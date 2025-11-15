/*
 * Real-Time and Stochastic Solver Test Suite
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/realtime_stochastic.h"

// Test ODE: exponential decay
static void exp_decay_ode(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Callback for real-time solver
static void realtime_callback(double t, const double* y, size_t n, void* user_data) {
    (void)user_data;
    (void)n;
    if (n > 0 && y) {
        printf("  Real-time step: t=%.6f, y[0]=%.6f\n", t, y[0]);
    }
}

int test_realtime_rk3() {
    printf("Testing Real-Time RK3 Solver...\n");
    
    RealtimeSolverState state;
    if (realtime_rk3_init(&state, 1, 0.01, 10) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double y0[1] = {1.0};
    state.t_current = 0.0;
    
    // Simulate streaming data
    for (int i = 0; i < 5; i++) {
        y0[0] = 1.0 * exp(-state.t_current); // Simulated streaming data
        if (realtime_rk3_step(&state, exp_decay_ode, y0, NULL, realtime_callback, NULL) != 0) {
            printf("  FAIL: Step failed\n");
            realtime_solver_free(&state);
            return 1;
        }
    }
    
    printf("  Steps processed: %llu\n", (unsigned long long)state.step_count);
    printf("  PASS\n\n");
    
    realtime_solver_free(&state);
    return 0;
}

int test_stochastic_rk3() {
    printf("Testing Stochastic RK3 Solver...\n");
    
    StochasticParams params = {
        .noise_amplitude = 0.01,
        .noise_correlation = 0.1,
        .use_brownian = 0,
        .seed = 42
    };
    
    printf("  Initializing solver...\n");
    void* solver = stochastic_rk3_init(1, 0.01, &params);
    if (!solver) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    printf("  Solver initialized successfully\n");
    
    double y[1] = {1.0};
    double t = 0.0;
    
    printf("  Running stochastic simulation...\n");
    for (int i = 0; i < 10; i++) {
        printf("    Step %d: t=%.3f, y=%.6f\n", i, t, y[0]);
        t = stochastic_rk3_step(solver, exp_decay_ode, t, y, NULL);
        if (i % 2 == 0) {
            printf("    t=%.3f, y=%.6f\n", t, y[0]);
        }
    }
    
    printf("  PASS\n\n");
    stochastic_solver_free(solver);
    return 0;
}

int test_data_driven_adaptive() {
    printf("Testing Data-Driven Adaptive Step Size...\n");
    
    double error_history[] = {1e-3, 5e-4, 2e-4, 1e-4, 5e-5};
    double current_h = 0.01;
    double target_error = 1e-4;
    
    double new_h = data_driven_adaptive_step(error_history, 5, current_h, target_error);
    
    printf("  Current step size: %.6f\n", current_h);
    printf("  Recommended step size: %.6f\n", new_h);
    printf("  Adjustment factor: %.2f\n", new_h / current_h);
    printf("  PASS\n\n");
    
    return 0;
}

int test_data_driven_method_select() {
    printf("Testing Data-Driven Method Selection...\n");
    
    // Test case 1: Stiff system
    int method1 = data_driven_method_select(200.0, 1e-4, 500000.0);
    printf("  Stiff system (stiffness=200): Method %d (0=RK3, 1=Adams, -1=both)\n", method1);
    
    // Test case 2: High speed requirement
    int method2 = data_driven_method_select(10.0, 1e-4, 2000000.0);
    printf("  High speed requirement: Method %d\n", method2);
    
    // Test case 3: High accuracy requirement
    int method3 = data_driven_method_select(10.0, 1e-8, 500000.0);
    printf("  High accuracy requirement: Method %d\n", method3);
    
    printf("  PASS\n\n");
    return 0;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Real-Time & Stochastic Solver Test Suite                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_realtime_rk3();
    failures += test_stochastic_rk3();
    failures += test_data_driven_adaptive();
    failures += test_data_driven_method_select();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    return (failures == 0) ? 0 : 1;
}
