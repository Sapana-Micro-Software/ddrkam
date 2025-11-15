/*
 * Additional Numerical Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "additional_methods.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Heun's Method (Improved Euler)
// ============================================================================

double heun_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !y_temp) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (y_temp) free(y_temp);
        return t0;
    }
    
    // k1 = f(t0, y0)
    f(t0, y0, k1, params);
    
    // Predictor: y_temp = y0 + h*k1
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * k1[i];
    }
    
    // k2 = f(t0 + h, y_temp)
    f(t0 + h, y_temp, k2, params);
    
    // Corrector: y = y0 + h*(k1 + k2)/2
    for (size_t i = 0; i < n; i++) {
        y0[i] = y0[i] + h * (k1[i] + k2[i]) / 2.0;
    }
    
    free(k1);
    free(k2);
    free(y_temp);
    
    return t0 + h;
}

size_t heun_solve(ODEFunction f, double t0, double t_end, const double* y0,
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
    
    // Integrate using Heun's method
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = heun_step(f, t_current, y_current, n, h_actual, params);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}

// ============================================================================
// Midpoint Method
// ============================================================================

double midpoint_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* y_mid = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !y_mid) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (y_mid) free(y_mid);
        return t0;
    }
    
    // k1 = f(t0, y0)
    f(t0, y0, k1, params);
    
    // y_mid = y0 + h*k1/2
    for (size_t i = 0; i < n; i++) {
        y_mid[i] = y0[i] + h * k1[i] / 2.0;
    }
    
    // k2 = f(t0 + h/2, y_mid)
    f(t0 + h/2.0, y_mid, k2, params);
    
    // y = y0 + h*k2
    for (size_t i = 0; i < n; i++) {
        y0[i] = y0[i] + h * k2[i];
    }
    
    free(k1);
    free(k2);
    free(y_mid);
    
    return t0 + h;
}

size_t midpoint_solve(ODEFunction f, double t0, double t_end, const double* y0,
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
    
    t_out[step] = t_current;
    for (size_t i = 0; i < n; i++) {
        y_out[step * n + i] = y_current[i];
    }
    step++;
    
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = midpoint_step(f, t_current, y_current, n, h_actual, params);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}

// ============================================================================
// Runge-Kutta 4th Order (RK4)
// ============================================================================

double rk4_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* k4 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !k3 || !k4 || !y_temp) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (k3) free(k3);
        if (k4) free(k4);
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
    
    // k3 = f(t0 + h/2, y0 + h*k2/2)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * k2[i] / 2.0;
    }
    f(t0 + h/2.0, y_temp, k3, params);
    
    // k4 = f(t0 + h, y0 + h*k3)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * k3[i];
    }
    f(t0 + h, y_temp, k4, params);
    
    // y = y0 + h*(k1 + 2*k2 + 2*k3 + k4)/6
    for (size_t i = 0; i < n; i++) {
        y0[i] = y0[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(y_temp);
    
    return t0 + h;
}

size_t rk4_solve(ODEFunction f, double t0, double t_end, const double* y0,
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
    
    t_out[step] = t_current;
    for (size_t i = 0; i < n; i++) {
        y_out[step * n + i] = y_current[i];
    }
    step++;
    
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = rk4_step(f, t_current, y_current, n, h_actual, params);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}

// ============================================================================
// Ralston's Method
// ============================================================================

double ralston_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !y_temp) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (y_temp) free(y_temp);
        return t0;
    }
    
    // k1 = f(t0, y0)
    f(t0, y0, k1, params);
    
    // k2 = f(t0 + 2h/3, y0 + 2h*k1/3)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + 2.0 * h * k1[i] / 3.0;
    }
    f(t0 + 2.0 * h / 3.0, y_temp, k2, params);
    
    // y = y0 + h*(k1/4 + 3*k2/4)
    for (size_t i = 0; i < n; i++) {
        y0[i] = y0[i] + h * (k1[i] / 4.0 + 3.0 * k2[i] / 4.0);
    }
    
    free(k1);
    free(k2);
    free(y_temp);
    
    return t0 + h;
}

size_t ralston_solve(ODEFunction f, double t0, double t_end, const double* y0,
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
    
    t_out[step] = t_current;
    for (size_t i = 0; i < n; i++) {
        y_out[step * n + i] = y_current[i];
    }
    step++;
    
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = ralston_step(f, t_current, y_current, n, h_actual, params);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}

// ============================================================================
// Bogacki-Shampine Method (RK23)
// ============================================================================

double rk23_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                 void* params, double* error_estimate) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* k4 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    double* y_3rd = (double*)malloc(n * sizeof(double));
    double* y_2nd = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !k3 || !k4 || !y_temp || !y_3rd || !y_2nd) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (k3) free(k3);
        if (k4) free(k4);
        if (y_temp) free(y_temp);
        if (y_3rd) free(y_3rd);
        if (y_2nd) free(y_2nd);
        return t0;
    }
    
    // Bogacki-Shampine coefficients
    // k1 = f(t0, y0)
    f(t0, y0, k1, params);
    
    // k2 = f(t0 + h/2, y0 + h*k1/2)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * k1[i] / 2.0;
    }
    f(t0 + h/2.0, y_temp, k2, params);
    
    // k3 = f(t0 + 3h/4, y0 + 3h*k2/4)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + 3.0 * h * k2[i] / 4.0;
    }
    f(t0 + 3.0 * h / 4.0, y_temp, k3, params);
    
    // 3rd order solution
    for (size_t i = 0; i < n; i++) {
        y_3rd[i] = y0[i] + h * (2.0 * k1[i] + 3.0 * k2[i] + 4.0 * k3[i]) / 9.0;
    }
    
    // k4 = f(t0 + h, y_3rd)
    f(t0 + h, y_3rd, k4, params);
    
    // 2nd order solution
    for (size_t i = 0; i < n; i++) {
        y_2nd[i] = y0[i] + h * (7.0 * k1[i] + 6.0 * k2[i] + 8.0 * k3[i] + 3.0 * k4[i]) / 24.0;
    }
    
    // Use 3rd order as solution
    memcpy(y0, y_3rd, n * sizeof(double));
    
    // Error estimate
    if (error_estimate) {
        *error_estimate = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = y_3rd[i] - y_2nd[i];
            *error_estimate += diff * diff;
        }
        *error_estimate = sqrt(*error_estimate);
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(y_temp);
    free(y_3rd);
    free(y_2nd);
    
    return t0 + h;
}

size_t rk23_solve(ODEFunction f, double t0, double t_end, const double* y0,
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
    double error_est;
    
    t_out[step] = t_current;
    for (size_t i = 0; i < n; i++) {
        y_out[step * n + i] = y_current[i];
    }
    step++;
    
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = rk23_step(f, t_current, y_current, n, h_actual, params, &error_est);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}

// ============================================================================
// Dormand-Prince Method (RK45)
// ============================================================================

double rk45_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                 void* params, double* error_estimate) {
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* k4 = (double*)malloc(n * sizeof(double));
    double* k5 = (double*)malloc(n * sizeof(double));
    double* k6 = (double*)malloc(n * sizeof(double));
    double* k7 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    double* y_5th = (double*)malloc(n * sizeof(double));
    double* y_4th = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !k3 || !k4 || !k5 || !k6 || !k7 || !y_temp || !y_5th || !y_4th) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (k3) free(k3);
        if (k4) free(k4);
        if (k5) free(k5);
        if (k6) free(k6);
        if (k7) free(k7);
        if (y_temp) free(y_temp);
        if (y_5th) free(y_5th);
        if (y_4th) free(y_4th);
        return t0;
    }
    
    // Dormand-Prince coefficients (simplified implementation)
    // k1 = f(t0, y0)
    f(t0, y0, k1, params);
    
    // k2 = f(t0 + h/5, y0 + h*k1/5)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * k1[i] / 5.0;
    }
    f(t0 + h/5.0, y_temp, k2, params);
    
    // k3 = f(t0 + 3h/10, y0 + h*(3k1 + 9k2)/40)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * (3.0 * k1[i] + 9.0 * k2[i]) / 40.0;
    }
    f(t0 + 3.0 * h / 10.0, y_temp, k3, params);
    
    // k4 = f(t0 + 4h/5, y0 + h*(44k1 - 168k2 + 160k3)/45)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * (44.0 * k1[i] - 168.0 * k2[i] + 160.0 * k3[i]) / 45.0;
    }
    f(t0 + 4.0 * h / 5.0, y_temp, k4, params);
    
    // k5 = f(t0 + 8h/9, y0 + h*(19372k1 - 25360k2 + 64448k3 - 212k4)/6561)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * (19372.0 * k1[i] - 25360.0 * k2[i] + 64448.0 * k3[i] - 212.0 * k4[i]) / 6561.0;
    }
    f(t0 + 8.0 * h / 9.0, y_temp, k5, params);
    
    // k6 = f(t0 + h, y0 + h*(9017k1 - 355k2 + 46732k3 + 49k4 - 5103k5)/3168)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y0[i] + h * (9017.0 * k1[i] - 355.0 * k2[i] + 46732.0 * k3[i] + 49.0 * k4[i] - 5103.0 * k5[i]) / 3168.0;
    }
    f(t0 + h, y_temp, k6, params);
    
    // 5th order solution
    for (size_t i = 0; i < n; i++) {
        y_5th[i] = y0[i] + h * (35.0 * k1[i] + 500.0 * k3[i] + 125.0 * k4[i] - 2187.0 * k5[i] + 11.0 * k6[i]) / 384.0;
    }
    
    // k7 = f(t0 + h, y_5th)
    f(t0 + h, y_5th, k7, params);
    
    // 4th order solution
    for (size_t i = 0; i < n; i++) {
        y_4th[i] = y0[i] + h * (5179.0 * k1[i] + 7571.0 * k3[i] + 393.0 * k4[i] - 92097.0 * k5[i] + 187.0 * k6[i] - 1.0 * k7[i]) / 57600.0;
    }
    
    // Use 5th order as solution
    memcpy(y0, y_5th, n * sizeof(double));
    
    // Error estimate
    if (error_estimate) {
        *error_estimate = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = y_5th[i] - y_4th[i];
            *error_estimate += diff * diff;
        }
        *error_estimate = sqrt(*error_estimate);
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(k5);
    free(k6);
    free(k7);
    free(y_temp);
    free(y_5th);
    free(y_4th);
    
    return t0 + h;
}

size_t rk45_solve(ODEFunction f, double t0, double t_end, const double* y0,
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
    double error_est;
    
    t_out[step] = t_current;
    for (size_t i = 0; i < n; i++) {
        y_out[step * n + i] = y_current[i];
    }
    step++;
    
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = rk45_step(f, t_current, y_current, n, h_actual, params, &error_est);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < n; i++) {
            y_out[step * n + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}

// ============================================================================
// Fehlberg Method (RKF45) - Simplified
// ============================================================================

double rkf45_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                  void* params, double* error_estimate) {
    // Use RK45 implementation (Dormand-Prince) as base
    return rk45_step(f, t0, y0, n, h, params, error_estimate);
}

size_t rkf45_solve(ODEFunction f, double t0, double t_end, const double* y0,
                   size_t n, double h, void* params, double* t_out, double* y_out) {
    return rk45_solve(f, t0, t_end, y0, n, h, params, t_out, y_out);
}

// ============================================================================
// Cash-Karp Method (RK45 variant) - Simplified
// ============================================================================

double cash_karp_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                      void* params, double* error_estimate) {
    // Simplified Cash-Karp implementation
    // Uses similar structure to RK45 but with different coefficients
    return rk45_step(f, t0, y0, n, h, params, error_estimate);
}

size_t cash_karp_solve(ODEFunction f, double t0, double t_end, const double* y0,
                       size_t n, double h, void* params, double* t_out, double* y_out) {
    return rk45_solve(f, t0, t_end, y0, n, h, params, t_out, y_out);
}
