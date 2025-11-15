/*
 * Adams Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "adams.h"
#include <stdlib.h>
#include <string.h>

void adams_bashforth3(ODEFunction f, const double* t, const double* y, 
                      size_t n, double h, void* params, double* y_pred) {
    double* f0 = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    double* f2 = (double*)malloc(n * sizeof(double));
    
    if (!f0 || !f1 || !f2) {
        if (f0) free(f0);
        if (f1) free(f1);
        if (f2) free(f2);
        return;
    }
    
    // Compute derivatives at previous points
    f(t[0], &y[0 * n], f0, params);  // f(t_n-2, y_n-2)
    f(t[1], &y[1 * n], f1, params);  // f(t_n-1, y_n-1)
    f(t[2], &y[2 * n], f2, params);  // f(t_n, y_n)
    
    // Adams-Bashforth 3rd order: y_n+1 = y_n + h*(23*f_n - 16*f_n-1 + 5*f_n-2)/12
    for (size_t i = 0; i < n; i++) {
        y_pred[i] = y[2 * n + i] + h * (23.0 * f2[i] - 16.0 * f1[i] + 5.0 * f0[i]) / 12.0;
    }
    
    free(f0);
    free(f1);
    free(f2);
}

void adams_moulton3(ODEFunction f, const double* t, const double* y, 
                    size_t n, double h, void* params, 
                    const double* y_pred, double* y_corr) {
    double* f_pred = (double*)malloc(n * sizeof(double));
    double* f0 = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    
    if (!f_pred || !f0 || !f1) {
        if (f_pred) free(f_pred);
        if (f0) free(f0);
        if (f1) free(f1);
        return;
    }
    
    double t_next = t[2] + h;
    
    // Compute derivatives
    f(t_next, y_pred, f_pred, params);  // f(t_n+1, y_pred)
    f(t[1], &y[1 * n], f0, params);     // f(t_n-1, y_n-1)
    f(t[2], &y[2 * n], f1, params);     // f(t_n, y_n)
    
    // Adams-Moulton 3rd order: y_n+1 = y_n + h*(5*f_n+1 + 8*f_n - f_n-1)/12
    for (size_t i = 0; i < n; i++) {
        y_corr[i] = y[2 * n + i] + h * (5.0 * f_pred[i] + 8.0 * f1[i] - f0[i]) / 12.0;
    }
    
    free(f_pred);
    free(f0);
    free(f1);
}
