#include "dgesv.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


double absVal(const double val) {
    if (val < 0) {
        return -val;
    } else {
        return val;
    }
}

void rowSwapper(double *matrix, const int row, const int new_row, const int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double temp = matrix[row * n + i];
        matrix[row * n + i] = matrix[new_row * n + i];
        matrix[new_row * n + i] = temp;
    }
}

void rowSubstract(double *matrix, const int row_result, const int row, const double ratio, const int n) {
    #pragma ivdep
    for (int i = 0; i < n; i++)
        matrix[row_result * n + i] -= ratio * matrix[row * n + i];
}

int gaussElimination(double *matrix, double *aug_matrix, const int n) {
    for (int i = 0; i < n; i++) {
        if (matrix[i * n + i] == 0) {
            int biggest = i;
            #pragma omp parallel for
            for (int j = 0; j < n; j++) {
                double actual = matrix[j * n + i];
                double actual_max = matrix[biggest * n + i];
                if (absVal(actual) > absVal(actual_max)) {
                    biggest = j;
                }
            }
            if (biggest != i) {
                rowSwapper(matrix, i, biggest, n);
                rowSwapper(aug_matrix, i, biggest, n);
            } else {
                return -1;
            }
        }
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double ratio = matrix[j * n + i] / matrix[i * n + i];
                if (ratio != 0) {
                    rowSubstract(matrix, j, i, ratio, n);
                    rowSubstract(aug_matrix, j, i, ratio, n);
                }
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug_matrix[i * n + j] /= matrix[i * n + i];
        }
    }
    return 0;
}

int my_dgesv(const int n, double *a, double *b) {
    if (gaussElimination(a, b, n) == -1) {
        printf("Singular matrix\n");
        return -1;
    }
    return 0;
}
