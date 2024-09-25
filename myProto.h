#pragma once

#define PART  1000

double* GPUGetXY(double *data, int n, double t);
int* GPUGetPoints(double *allValuesX, double *allValuesY, int sizeArr, double D, int K, int StartIndex);



