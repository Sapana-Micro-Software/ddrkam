/*
 * Test suite for Hierarchical RK method
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/hierarchical_rk.h"

void test_ode(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
    dydt[1] = -2.0 * y[1];
}

int main() {
    printf("=== Hierarchical RK Test ===\n\n");
    
    HierarchicalRKSolver solver;
    if (hierarchical_rk_init(&solver, 3, 2, 16) != 0) {
        printf("Failed to initialize solver\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 1.0;
    double y0[2] = {1.0, 1.0};
    double h = 0.01;
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * 2 * sizeof(double));
    
    size_t num_steps = hierarchical_rk_solve(&solver, test_ode, t0, t_end, y0, h, NULL, t_out, y_out);
    
    printf("Steps: %zu\n", num_steps);
    printf("Final state: [%.6f, %.6f]\n", y_out[(num_steps-1)*2], y_out[(num_steps-1)*2+1]);
    printf("Expected: [%.6f, %.6f]\n", exp(-1.0), exp(-2.0));
    
    hierarchical_rk_free(&solver);
    free(t_out);
    free(y_out);
    
    printf("PASS\n");
    return 0;
}
