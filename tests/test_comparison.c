/*
 * Comparison Test Suite
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/comparison.h"

// Test ODE: dy/dt = -y, solution: y(t) = y0 * exp(-t)
void test_exponential(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

// Test ODE: Simple harmonic oscillator
void test_oscillator(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = y[1];
    dydt[1] = -y[0];
}

int test_exponential_comparison() {
    printf("=== Exponential Decay Comparison ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0;
    double y0[1] = {1.0};
    double h = 0.01;
    double exact[1] = {exp(-t_end)};
    
    ComparisonResults results;
    
    if (compare_methods(test_exponential, t0, t_end, y0, 1, h, NULL, exact, &results) != 0) {
        printf("Comparison failed\n");
        return 1;
    }
    
    print_comparison_results(&results);
    export_comparison_csv("exponential_comparison.csv", &results);
    
    return 0;
}

int test_oscillator_comparison() {
    printf("=== Harmonic Oscillator Comparison ===\n");
    
    double t0 = 0.0;
    double t_end = 2.0 * M_PI;
    double y0[2] = {1.0, 0.0};
    double h = 0.01;
    double exact[2] = {cos(t_end), -sin(t_end)};
    
    ComparisonResults results;
    
    if (compare_methods(test_oscillator, t0, t_end, y0, 2, h, NULL, exact, &results) != 0) {
        printf("Comparison failed\n");
        return 1;
    }
    
    print_comparison_results(&results);
    export_comparison_csv("oscillator_comparison.csv", &results);
    
    return 0;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  DDRKAM Method Comparison: RK3 vs DDRK3 vs AM vs DDAM       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_exponential_comparison();
    failures += test_oscillator_comparison();
    
    printf("=== Comparison Complete ===\n");
    printf("CSV files exported: exponential_comparison.csv, oscillator_comparison.csv\n");
    
    return (failures == 0) ? 0 : 1;
}
