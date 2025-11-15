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

// ============================================================================
// Adams-Bashforth 1st Order (Euler's Method)
// ============================================================================

void adams_bashforth1(ODEFunction f, const double* t, const double* y, 
                      size_t n, double h, void* params, double* y_pred) {
    double* f0 = (double*)malloc(n * sizeof(double));
    
    if (!f0) return;
    
    // Compute derivative at current point
    f(t[0], &y[0 * n], f0, params);  // f(t_n, y_n)
    
    // Adams-Bashforth 1st order (Euler): y_n+1 = y_n + h*f_n
    for (size_t i = 0; i < n; i++) {
        y_pred[i] = y[0 * n + i] + h * f0[i];
    }
    
    free(f0);
}

// ============================================================================
// Adams-Moulton 1st Order (Implicit Euler)
// ============================================================================

void adams_moulton1(ODEFunction f, const double* t, const double* y, 
                    size_t n, double h, void* params, 
                    const double* y_pred, double* y_corr) {
    double* f_pred = (double*)malloc(n * sizeof(double));
    
    if (!f_pred) return;
    
    double t_next = t[0] + h;
    
    // Compute derivative at predicted point
    f(t_next, y_pred, f_pred, params);  // f(t_n+1, y_pred)
    
    // Adams-Moulton 1st order (Implicit Euler): y_n+1 = y_n + h*f_n+1
    for (size_t i = 0; i < n; i++) {
        y_corr[i] = y[0 * n + i] + h * f_pred[i];
    }
    
    free(f_pred);
}

// ============================================================================
// Adams-Bashforth 2nd Order
// ============================================================================

void adams_bashforth2(ODEFunction f, const double* t, const double* y, 
                      size_t n, double h, void* params, double* y_pred) {
    double* f0 = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    
    if (!f0 || !f1) {
        if (f0) free(f0);
        if (f1) free(f1);
        return;
    }
    
    // Compute derivatives at previous points
    f(t[0], &y[0 * n], f0, params);  // f(t_n-1, y_n-1)
    f(t[1], &y[1 * n], f1, params);  // f(t_n, y_n)
    
    // Adams-Bashforth 2nd order: y_n+1 = y_n + h*(3*f_n - f_n-1)/2
    for (size_t i = 0; i < n; i++) {
        y_pred[i] = y[1 * n + i] + h * (3.0 * f1[i] - f0[i]) / 2.0;
    }
    
    free(f0);
    free(f1);
}

// ============================================================================
// Adams-Moulton 2nd Order (Trapezoidal Rule)
// ============================================================================

void adams_moulton2(ODEFunction f, const double* t, const double* y, 
                    size_t n, double h, void* params, 
                    const double* y_pred, double* y_corr) {
    double* f_pred = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    
    if (!f_pred || !f1) {
        if (f_pred) free(f_pred);
        if (f1) free(f1);
        return;
    }
    
    double t_next = t[1] + h;
    
    // Compute derivatives
    f(t_next, y_pred, f_pred, params);  // f(t_n+1, y_pred)
    f(t[1], &y[1 * n], f1, params);     // f(t_n, y_n)
    
    // Adams-Moulton 2nd order (Trapezoidal): y_n+1 = y_n + h*(f_n+1 + f_n)/2
    for (size_t i = 0; i < n; i++) {
        y_corr[i] = y[1 * n + i] + h * (f_pred[i] + f1[i]) / 2.0;
    }
    
    free(f_pred);
    free(f1);
}

// ============================================================================
// Adams-Bashforth 4th Order
// ============================================================================

void adams_bashforth4(ODEFunction f, const double* t, const double* y, 
                      size_t n, double h, void* params, double* y_pred) {
    double* f0 = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    double* f2 = (double*)malloc(n * sizeof(double));
    double* f3 = (double*)malloc(n * sizeof(double));
    
    if (!f0 || !f1 || !f2 || !f3) {
        if (f0) free(f0);
        if (f1) free(f1);
        if (f2) free(f2);
        if (f3) free(f3);
        return;
    }
    
    // Compute derivatives at previous points
    f(t[0], &y[0 * n], f0, params);  // f(t_n-3, y_n-3)
    f(t[1], &y[1 * n], f1, params);  // f(t_n-2, y_n-2)
    f(t[2], &y[2 * n], f2, params);  // f(t_n-1, y_n-1)
    f(t[3], &y[3 * n], f3, params);  // f(t_n, y_n)
    
    // Adams-Bashforth 4th order: y_n+1 = y_n + h*(55*f_n - 59*f_n-1 + 37*f_n-2 - 9*f_n-3)/24
    for (size_t i = 0; i < n; i++) {
        y_pred[i] = y[3 * n + i] + h * (55.0 * f3[i] - 59.0 * f2[i] + 37.0 * f1[i] - 9.0 * f0[i]) / 24.0;
    }
    
    free(f0);
    free(f1);
    free(f2);
    free(f3);
}

// ============================================================================
// Adams-Moulton 4th Order
// ============================================================================

void adams_moulton4(ODEFunction f, const double* t, const double* y, 
                    size_t n, double h, void* params, 
                    const double* y_pred, double* y_corr) {
    double* f_pred = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    double* f2 = (double*)malloc(n * sizeof(double));
    double* f3 = (double*)malloc(n * sizeof(double));
    
    if (!f_pred || !f1 || !f2 || !f3) {
        if (f_pred) free(f_pred);
        if (f1) free(f1);
        if (f2) free(f2);
        if (f3) free(f3);
        return;
    }
    
    double t_next = t[3] + h;
    
    // Compute derivatives
    f(t_next, y_pred, f_pred, params);  // f(t_n+1, y_pred)
    f(t[1], &y[1 * n], f1, params);     // f(t_n-2, y_n-2)
    f(t[2], &y[2 * n], f2, params);     // f(t_n-1, y_n-1)
    f(t[3], &y[3 * n], f3, params);     // f(t_n, y_n)
    
    // Adams-Moulton 4th order: y_n+1 = y_n + h*(9*f_n+1 + 19*f_n - 5*f_n-1 + f_n-2)/24
    for (size_t i = 0; i < n; i++) {
        y_corr[i] = y[3 * n + i] + h * (9.0 * f_pred[i] + 19.0 * f3[i] - 5.0 * f2[i] + f1[i]) / 24.0;
    }
    
    free(f_pred);
    free(f1);
    free(f2);
    free(f3);
}

// ============================================================================
// Adams-Bashforth 5th Order
// ============================================================================

void adams_bashforth5(ODEFunction f, const double* t, const double* y, 
                      size_t n, double h, void* params, double* y_pred) {
    double* f0 = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    double* f2 = (double*)malloc(n * sizeof(double));
    double* f3 = (double*)malloc(n * sizeof(double));
    double* f4 = (double*)malloc(n * sizeof(double));
    
    if (!f0 || !f1 || !f2 || !f3 || !f4) {
        if (f0) free(f0);
        if (f1) free(f1);
        if (f2) free(f2);
        if (f3) free(f3);
        if (f4) free(f4);
        return;
    }
    
    // Compute derivatives at previous points
    f(t[0], &y[0 * n], f0, params);  // f(t_n-4, y_n-4)
    f(t[1], &y[1 * n], f1, params);  // f(t_n-3, y_n-3)
    f(t[2], &y[2 * n], f2, params);  // f(t_n-2, y_n-2)
    f(t[3], &y[3 * n], f3, params);  // f(t_n-1, y_n-1)
    f(t[4], &y[4 * n], f4, params);  // f(t_n, y_n)
    
    // Adams-Bashforth 5th order: y_n+1 = y_n + h*(1901*f_n - 2774*f_n-1 + 2616*f_n-2 - 1274*f_n-3 + 251*f_n-4)/720
    for (size_t i = 0; i < n; i++) {
        y_pred[i] = y[4 * n + i] + h * (1901.0 * f4[i] - 2774.0 * f3[i] + 2616.0 * f2[i] - 1274.0 * f1[i] + 251.0 * f0[i]) / 720.0;
    }
    
    free(f0);
    free(f1);
    free(f2);
    free(f3);
    free(f4);
}

// ============================================================================
// Adams-Moulton 5th Order
// ============================================================================

void adams_moulton5(ODEFunction f, const double* t, const double* y, 
                    size_t n, double h, void* params, 
                    const double* y_pred, double* y_corr) {
    double* f_pred = (double*)malloc(n * sizeof(double));
    double* f1 = (double*)malloc(n * sizeof(double));
    double* f2 = (double*)malloc(n * sizeof(double));
    double* f3 = (double*)malloc(n * sizeof(double));
    double* f4 = (double*)malloc(n * sizeof(double));
    
    if (!f_pred || !f1 || !f2 || !f3 || !f4) {
        if (f_pred) free(f_pred);
        if (f1) free(f1);
        if (f2) free(f2);
        if (f3) free(f3);
        if (f4) free(f4);
        return;
    }
    
    double t_next = t[4] + h;
    
    // Compute derivatives
    f(t_next, y_pred, f_pred, params);  // f(t_n+1, y_pred)
    f(t[1], &y[1 * n], f1, params);     // f(t_n-3, y_n-3)
    f(t[2], &y[2 * n], f2, params);     // f(t_n-2, y_n-2)
    f(t[3], &y[3 * n], f3, params);     // f(t_n-1, y_n-1)
    f(t[4], &y[4 * n], f4, params);     // f(t_n, y_n)
    
    // Adams-Moulton 5th order: y_n+1 = y_n + h*(251*f_n+1 + 646*f_n - 264*f_n-1 + 106*f_n-2 - 19*f_n-3)/720
    for (size_t i = 0; i < n; i++) {
        y_corr[i] = y[4 * n + i] + h * (251.0 * f_pred[i] + 646.0 * f4[i] - 264.0 * f3[i] + 106.0 * f2[i] - 19.0 * f1[i]) / 720.0;
    }
    
    free(f_pred);
    free(f1);
    free(f2);
    free(f3);
    free(f4);
}
