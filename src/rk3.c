/*
 * Runge-Kutta 3rd Order Method Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "rk3.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

double rk3_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !k3 || !y_temp) {
        // Memory allocation failed
        if (k1) free(k1);
        if (k2) free(k2);
        if (k3) free(k3);
        if (y_temp) free(y_temp);
        return t0;
    }
    
    // k1 = f(t0, y0)
    f(t0, y0, k1, params);
    
    // k2 = f(t0 + h/2, y0 + h*k1/2)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * k1[i] / 2.0;
    }
    f(t0 + h/2.0, y_temp, k2, params);
    
    // k3 = f(t0 + h, y0 - h*k1 + 2*h*k2)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] - h * k1[i] + 2.0 * h * k2[i];
    }
    f(t0 + h, y_temp, k3, params);
    
    // y_new = y0 + h*(k1 + 4*k2 + k3)/6
    for (size_t i = 0; i < n; i++) {
        y0[i] = y0[i] + h * (k1[i] + 4.0 * k2[i] + k3[i]) / 6.0;
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(y_temp);
    
    return t0 + h;
}

size_t rk3_solve(ODEFunction f, double t0, double t_end, const double* y0, 
                 size_t n, double h, void* params, double* t_out, double* y_out) {
    if (h <= 0.0 || t_end <= t0) {
        return 0;
    }
    
    double* y_current = (double*)malloc(n * sizeof(double));
    if (!y_current) {
        return 0;
    }
    
    memcpy(y_current, y0, n * sizeof(double));
    
    double t_current = t0;
    size_t step = 0;
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    
    // Store initial condition
    if (t_out) t_out[step] = t_current;
    if (y_out) {
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
    }
    step++;
    
    // Integrate
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = rk3_step(f, t_current, y_current, n, h_actual, params);
        
        if (t_out) t_out[step] = t_current;
        if (y_out) {
            for (size_t i = 0; i < n; i++) {
                y_out[step * n + i] = y_current[i];
            }
        }
        step++;
    }
    
    free(y_current);
    return step;
}
