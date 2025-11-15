/*
 * Test suite for Runge-Kutta 3rd order method
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/rk3.h"

// Test ODE: dy/dt = -y, solution: y(t) = y0 * exp(-t)
void test_exponential_decay(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Test ODE: Lorenz system
void lorenz_ode(double t, const double* y, double* dydt, void* params) {
    (void)t;
    double* p = (double*)params;
    double sigma = p[0];
    double rho = p[1];
    double beta = p[2];
    
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

int test_exponential() {
    printf("Testing exponential decay ODE...\n");
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * sizeof(double));
    
    size_t num_steps = rk3_solve(test_exponential_decay, t0, t_end, y0, 1, h, NULL, t_out, y_out);
    
    double error = fabs(y_out[num_steps-1] - exp(-t_end));
    printf("  Final value: %.6f, Expected: %.6f, Error: %.6e\n", 
           y_out[num_steps-1], exp(-t_end), error);
    
    int success = (error < 1e-3);
    printf("  %s\n\n", success ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return success ? 0 : 1;
}

int test_lorenz() {
    printf("Testing Lorenz system...\n");
    
    double params[3] = {10.0, 28.0, 8.0/3.0};
    double t0 = 0.0;
    double t_end = 1.0;
    double y0[3] = {1.0, 1.0, 1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * 3 * sizeof(double));
    
    size_t num_steps = rk3_solve(lorenz_ode, t0, t_end, y0, 3, h, params, t_out, y_out);
    
    printf("  Steps: %zu\n", num_steps);
    printf("  Final state: [%.6f, %.6f, %.6f]\n", 
           y_out[(num_steps-1)*3], y_out[(num_steps-1)*3+1], y_out[(num_steps-1)*3+2]);
    
    // Check that solution is bounded (Lorenz should stay bounded)
    double norm = sqrt(y_out[(num_steps-1)*3]*y_out[(num_steps-1)*3] +
                       y_out[(num_steps-1)*3+1]*y_out[(num_steps-1)*3+1] +
                       y_out[(num_steps-1)*3+2]*y_out[(num_steps-1)*3+2]);
    int success = (norm < 100.0); // Reasonable bound
    printf("  Final norm: %.6f\n", norm);
    printf("  %s\n\n", success ? "PASS" : "FAIL");
    
    free(t_out);
    free(y_out);
    return success ? 0 : 1;
}

int main() {
    printf("=== Runge-Kutta 3rd Order Test Suite ===\n\n");
    
    int failures = 0;
    failures += test_exponential();
    failures += test_lorenz();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    return (failures == 0) ? 0 : 1;
}
