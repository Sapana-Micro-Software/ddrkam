/*
 * Test Suite for Interior Point Methods
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/nonlinear_solver.h"

// Test ODE: dy/dt = -y (exponential decay)
void test_ode(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Objective function: minimize ||dy/dt - f(t,y)||^2
double objective_function(const double* x, size_t n, void* params) {
    (void)params;
    double* dydt = (double*)malloc(n * sizeof(double));
    if (!dydt) return 1e10;
    
    test_ode(0.0, x, dydt, NULL);
    
    // For exponential decay: we want to minimize the residual
    double residual = 0.0;
    for (size_t i = 0; i < n; i++) {
        double expected = -x[i]; // dy/dt = -y
        residual += (dydt[i] - expected) * (dydt[i] - expected);
    }
    
    free(dydt);
    return residual;
}

// Constraint function: y >= 0 (non-negativity constraint)
void constraint_function(const double* x, size_t n, double* constraints, void* params) {
    (void)params;
    for (size_t i = 0; i < n; i++) {
        constraints[i] = -x[i]; // g(x) = -x <= 0 means x >= 0
    }
}

int test_interior_point_basic() {
    printf("=== Test 1: Basic Interior Point Method ===\n");
    
    NonlinearODESolver solver;
    double y0[1] = {1.0};
    double y_out[1];
    
    // Initialize with interior point method
    if (nonlinear_ode_init(&solver, 1, NLP_INTERIOR_POINT,
                           objective_function, constraint_function, NULL) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    // Set constraints
    nonlinear_ode_set_constraints(&solver, 1);
    
    // Configure interior point parameters
    InteriorPointParams ip_params = {
        .barrier_parameter = 1.0,
        .barrier_reduction = 0.1,
        .centering_parameter = 0.1,
        .feasibility_tolerance = 1e-6,
        .optimality_tolerance = 1e-6,
        .max_barrier_iterations = 100,
        .handle_nonconvex = 0,
        .perturbation_radius = 0.01
    };
    nonlinear_ode_set_interior_point_params(&solver, &ip_params);
    
    // Solve
    if (nonlinear_ode_solve(&solver, test_ode, 0.0, 1.0, y0, y_out) != 0) {
        printf("  FAIL: Solve failed\n");
        nonlinear_ode_free(&solver);
        return 1;
    }
    
    printf("  Initial: y(0) = %.6f\n", y0[0]);
    printf("  Final: y(1) = %.6f\n", y_out[0]);
    printf("  Expected: y(1) = %.6f (exp(-1))\n", exp(-1.0));
    printf("  Error: %.6e\n", fabs(y_out[0] - exp(-1.0)));
    printf("  %s\n\n", (fabs(y_out[0] - exp(-1.0)) < 0.1) ? "PASS" : "FAIL");
    
    nonlinear_ode_free(&solver);
    return (fabs(y_out[0] - exp(-1.0)) < 0.1) ? 0 : 1;
}

int test_interior_point_nonconvex() {
    printf("=== Test 2: Non-Convex Interior Point Method ===\n");
    
    NonlinearODESolver solver;
    double y0[1] = {1.0};
    double y_out[1];
    
    if (nonlinear_ode_init(&solver, 1, NLP_INTERIOR_POINT,
                           objective_function, constraint_function, NULL) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    nonlinear_ode_set_constraints(&solver, 1);
    
    // Enable non-convex handling
    nonlinear_ode_enable_nonconvex(&solver, 1, 0.01);
    
    if (nonlinear_ode_solve(&solver, test_ode, 0.0, 1.0, y0, y_out) != 0) {
        printf("  FAIL: Solve failed\n");
        nonlinear_ode_free(&solver);
        return 1;
    }
    
    printf("  Initial: y(0) = %.6f\n", y0[0]);
    printf("  Final: y(1) = %.6f\n", y_out[0]);
    printf("  Expected: y(1) = %.6f\n", exp(-1.0));
    printf("  Error: %.6e\n", fabs(y_out[0] - exp(-1.0)));
    printf("  %s\n\n", (fabs(y_out[0] - exp(-1.0)) < 0.1) ? "PASS" : "FAIL");
    
    nonlinear_ode_free(&solver);
    return (fabs(y_out[0] - exp(-1.0)) < 0.1) ? 0 : 1;
}

int test_online_interior_point() {
    printf("=== Test 3: Online Interior Point Method ===\n");
    
    NonlinearODESolver base_solver;
    OnlineNonlinearSolver online_solver;
    double y0[1] = {1.0};
    double y_out[1];
    
    if (nonlinear_ode_init(&base_solver, 1, NLP_INTERIOR_POINT,
                           objective_function, constraint_function, NULL) != 0) {
        printf("  FAIL: Base solver initialization failed\n");
        return 1;
    }
    
    nonlinear_ode_set_constraints(&base_solver, 1);
    
    if (online_nonlinear_init(&online_solver, &base_solver, 0.01) != 0) {
        printf("  FAIL: Online solver initialization failed\n");
        nonlinear_ode_free(&base_solver);
        return 1;
    }
    
    if (online_nonlinear_solve(&online_solver, test_ode, 0.0, 1.0, y0, y_out) != 0) {
        printf("  FAIL: Solve failed\n");
        online_nonlinear_free(&online_solver);
        nonlinear_ode_free(&base_solver);
        return 1;
    }
    
    printf("  Initial: y(0) = %.6f\n", y0[0]);
    printf("  Final: y(1) = %.6f\n", y_out[0]);
    printf("  Expected: y(1) = %.6f\n", exp(-1.0));
    printf("  Error: %.6e\n", fabs(y_out[0] - exp(-1.0)));
    printf("  %s\n\n", (fabs(y_out[0] - exp(-1.0)) < 0.1) ? "PASS" : "FAIL");
    
    online_nonlinear_free(&online_solver);
    nonlinear_ode_free(&base_solver);
    return (fabs(y_out[0] - exp(-1.0)) < 0.1) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Interior Point Method Test Suite                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_interior_point_basic();
    failures += test_interior_point_nonconvex();
    failures += test_online_interior_point();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All Interior Point Method tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
