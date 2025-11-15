/*
 * Euler's Method Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

double euler_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params) {
    double* dydt = (double*)malloc(n * sizeof(double));
    
    if (!dydt) {
        return t0;
    }
    
    // Compute derivative: f(t0, y0)
    f(t0, y0, dydt, params);
    
    // Euler's method: y_{n+1} = y_n + h * f(t_n, y_n)
    for (size_t i = 0; i < n; i++) {
        y0[i] = y0[i] + h * dydt[i];
    }
    
    free(dydt);
    return t0 + h;
}

size_t euler_solve(ODEFunction f, double t0, double t_end, const double* y0,
                   size_t n, double h, void* params, double* t_out, double* y_out) {
    if (h <= 0.0 || t_end <= t0 || !f || !y0 || !t_out || !y_out) {
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
    t_out[step] = t_current;
    for (size_t i = 0; i < n; i++) {
        y_out[step * n + i] = y_current[i];
    }
    step++;
    
    // Integrate using Euler's method
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = euler_step(f, t_current, y_current, n, h_actual, params);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}
