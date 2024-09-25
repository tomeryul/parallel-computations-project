#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"
#include <time.h>
#include <math.h>



int main(int argc, char *argv[]) {
    int  my_rank; /* rank of process */
	int  p;       /* number of processes */
	int tag=0;    /* tag for messages */
	MPI_Status status ;   /* return status for receive */
	int *data,*splitResoultArr;

	int  sent, placeToAdd,TCount,K,resoultIndex;
	int numValues, start, end, tRange, howMuchToSend;
	double D;
	
	double t1,t2; // for the time measure

	double* matValues;
	

	/* start up MPI */
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p); 
  	
  	if (my_rank == 0) {
  	
  		t1 = MPI_Wtime();
  		FILE *file;
  		
    		char line[100];
    		
    		int  i;

    		// Open the file for reading
    		file = fopen("Input.txt", "r");
    		if (file == NULL) {
        		printf("Error opening the file.\n");
        		return 1;
    		}
		
		if(fscanf(file,"%d %d %lf %d\n",&numValues,&K,&D,&TCount)!= 4){
			printf("Error reading the file.\n");
        		return 1;
		}
		printf("numValues = %d,K = %d, D = %f, TCount = %d\n",numValues,K,D,TCount);
    		
		matValues = (double*)malloc(numValues*5 * sizeof(double));
		
    		// Read the remaining lines and store the values in the matValues
    		for (i = 0; i < numValues; i++) {
    			for(int j = 0 ; j < 5 ; j++){
    				fscanf(file, "%lf", &matValues[i*5+j]);
    			}
    			fgets(line, sizeof(line), file);	
    		}  

    		
    	}

		MPI_Bcast(&numValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&TCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

		double* allValuesX = (double*)malloc(numValues * sizeof(double));
		double* allValuesY = (double*)malloc(numValues * sizeof(double));
		double* returndXY = (double*)malloc(PART*2 * sizeof(double));
		int* returndAllPoints = (int*)malloc(3 * sizeof(int));
		int* returndPoint = (int*)malloc(3 * sizeof(int));

		if(my_rank != 0)
			matValues = (double*)malloc(numValues*5 * sizeof(double));

		MPI_Bcast(matValues, numValues*5, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		
		// calculating the range of i for each process 
		if((double)TCount/p == (int)TCount/p){
			if(my_rank == p-1){
				tRange = (TCount / p)+1;
				start = my_rank*(tRange-1);
				end = start+tRange;
				}
			else{
				tRange = TCount / p;
				start = my_rank*tRange;
				end = start+tRange;
				}
		}
		
		else{
			if(my_rank == p-1){
				tRange = (TCount - (p-1)*(int)(TCount/p))+1;
				start = TCount-(tRange-1);
				end = start+tRange;
				}
			else{
				tRange = (int)TCount/p;
				start = my_rank*tRange;
				end = start+tRange;
				}
		}
		

		
		// calculating the values for each process 

		// array of the resoult
		splitResoultArr = (int*)malloc(tRange*3 * sizeof(int));
		resoultIndex = 0;

		for (int i = start; i < end; i++){
			sent = 0; placeToAdd = 0;
			double t = 2*i/(double)TCount-1;
			do{
				// every time i calculate x,y for PART amount of points with CUDA
				howMuchToSend = (numValues-(sent/5));
				if(howMuchToSend > PART)
					howMuchToSend = PART;
				returndXY = GPUGetXY(matValues+sent, howMuchToSend*5,t); // PART is the num of treads in CUDA
				
				// adding the x,y of the new points to the array.
				for(int j = 0 ; j < howMuchToSend  ; j++){
    					allValuesX[placeToAdd] = returndXY[j];
    					allValuesY[placeToAdd] = returndXY[j+howMuchToSend];
    					placeToAdd += 1;
    				}
    					

				sent+= howMuchToSend*5;
					
					
					
			}while((numValues-(sent/5))>0);
				
			// after i have all the values (x,y) for all the point i use CUDA again to check 1000 points at the same time 
			// if they have k points that the distanse between them is lower than D.
				
			int addIn = 0,x = 0;
					
			while(x <= ceil(numValues/PART) && addIn < 3){
				returndPoint = GPUGetPoints(allValuesX, allValuesY, numValues, D, K, x);
						
				for(int j = 0 ; j < 3 && addIn < 3; j++){
					if(returndPoint[j] != -1){
						returndAllPoints[addIn] = returndPoint[j];
						addIn++;
					}
				}
						
				x++;
								
			}
				
			// if i dont have 3 points for this "t" i convert it to arr of -1.	
			if(addIn < 3){
				returndAllPoints[0] = -1;
				returndAllPoints[1] = -1;
				returndAllPoints[2] = -1;
			}
				
			// adding the resoult to the resoultArr
			for(int i = 0; i < 3 ; i++){
				splitResoultArr[resoultIndex] = returndAllPoints[i];
				resoultIndex += 1;
			}
				
		
				
		}
		
	
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		if(my_rank != 0){
			MPI_Send(&tRange, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(splitResoultArr, tRange*3, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		else{
		
		    	// i add 3 to the size because the t values is form -1 to 1 
    			int* finaleResoultArr = (int*)malloc((TCount*3+3) * sizeof(int));  
    			
    			
			int inTo = 0;
			int returndRange;
			for(int i = 0 ; i < tRange*3 ; i++){
				finaleResoultArr[inTo] = splitResoultArr[i];
				inTo++;
			}
			
			for(int i = 1 ; i < p ; i++){
				MPI_Recv(&returndRange, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(finaleResoultArr+inTo, returndRange*3, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				inTo+=returndRange*3;
			}
			
			// open the file for writing
   			FILE *file;
   			
   			file = fopen("Output.txt", "w");
    			if (file == NULL){
        			printf("Error opening the file");
        			return -1;
    			}
    			
    			// write to the text file
    			int endCount = 0;
    			#pragma omp parallel for reduction(+ : endCount)
    			for(int i = 0 ; i <= TCount*3 ; i+=3){
    				if(finaleResoultArr[i] != -1){
        				fprintf(file,"points %d,%d,%d satisfy Proximity Criteria at t = %lf \n",finaleResoultArr[i],finaleResoultArr[i+1],finaleResoultArr[i+2],((2*(i/3))/((double)TCount))-1);
        				endCount += 1;
        			}
        		}
        		
        		if(endCount == 0)
        			fprintf(file,"There were no 3 points found for any t.");
        			
        			
    			// close the file
    			fclose(file);
    			
    			// free allocated memory from process number 0
    			free(finaleResoultArr);
    			
    			//printing the run time
    			t2 = MPI_Wtime();
    			printf("MPI_Wtime measured a 1 second sleep to be: %1.2f\n",  t2-t1); fflush(stdout);

		}
		
		// free allocated memory from all processes
		free(allValuesX);
		free(allValuesY);
		free(returndXY);
		free(returndAllPoints);
		free(returndPoint);
		free(splitResoultArr);
    	

    MPI_Finalize();

    return 0;
}


