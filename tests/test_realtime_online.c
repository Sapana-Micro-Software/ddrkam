/*
 * Test suite for Real-Time, Online, and Dynamic Methods
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/realtime_online.h"

// Test ODE: dy/dt = -y
void test_exponential(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Callback for real-time processing
void realtime_callback(double t, const double* y, size_t n, void* user_data) {
    (void)user_data;
    // Can process data in real-time here
    (void)t;
    (void)y;
    (void)n;
}

int test_realtime_rk3() {
    printf("=== Real-Time RK3 Test ===\n");
    
    RealtimeRKSolver solver;
    if (realtime_rk_init(&solver, 1, 0.01, realtime_callback, NULL) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    
    clock_t start = clock();
    double t_current = t0;
    size_t steps = 0;
    double y[1] = {y0[0]};
    
    while (t_current < t_end && steps < 1000) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = realtime_rk_step(&solver, test_exponential, t_current, y, h_actual, NULL);
        steps++;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_exact = exp(-t_end);
    double error = fabs(y[0] - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y[0]);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  %s\n\n", (error < 1e-3) ? "PASS" : "FAIL");
    
    realtime_rk_free(&solver);
    return (error < 1e-3) ? 0 : 1;
}

int test_online_rk3() {
    printf("=== Online RK3 Test ===\n");
    
    OnlineRKSolver solver;
    if (online_rk_init(&solver, 1, 0.01, 0.01) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y[1] = {1.0};
    
    clock_t start = clock();
    double t_current = t0;
    size_t steps = 0;
    
    while (t_current < t_end && steps < 1000) {
        t_current = online_rk_step(&solver, test_exponential, t_current, y, NULL);
        steps++;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_exact = exp(-t_end);
    double error = fabs(y[0] - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y[0]);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  Adaptations: %zu\n", solver.adaptation_count);
    printf("  %s\n\n", (error < 1e-3) ? "PASS" : "FAIL");
    
    online_rk_free(&solver);
    return (error < 1e-3) ? 0 : 1;
}

int test_dynamic_rk3() {
    printf("=== Dynamic RK3 Test ===\n");
    
    DynamicRKSolver solver;
    if (dynamic_rk_init(&solver, 1, 0.01, 0.01) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y[1] = {1.0};
    
    clock_t start = clock();
    double t_current = t0;
    size_t steps = 0;
    
    while (t_current < t_end && steps < 1000) {
        t_current = dynamic_rk_step(&solver, test_exponential, t_current, y, NULL);
        steps++;
    }
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    double y_exact = exp(-t_end);
    double error = fabs(y[0] - y_exact);
    double accuracy = (1.0 - error / fabs(y_exact)) * 100.0;
    
    printf("  Steps: %zu\n", steps);
    printf("  Time: %.6f seconds\n", elapsed);
    printf("  Final value: %.10f\n", y[0]);
    printf("  Exact value: %.10f\n", y_exact);
    printf("  Error: %.6e\n", error);
    printf("  Accuracy: %.6f%%\n", accuracy);
    printf("  Final step size: %.6e\n", *solver.dynamic_step_size);
    printf("  %s\n\n", (error < 1e-3) ? "PASS" : "FAIL");
    
    dynamic_rk_free(&solver);
    return (error < 1e-3) ? 0 : 1;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Real-Time, Online, and Dynamic Methods Test Suite           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_realtime_rk3();
    failures += test_online_rk3();
    failures += test_dynamic_rk3();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    if (failures == 0) {
        printf("✅ All real-time/online/dynamic method tests PASSED\n");
    } else {
        printf("❌ Some tests FAILED\n");
    }
    
    return (failures == 0) ? 0 : 1;
}
