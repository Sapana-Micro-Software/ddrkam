/*
 * Comprehensive Benchmark Runner
 * Runs all methods and collects statistics
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
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

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Comprehensive Benchmark Suite - All Methods                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Test 1: Exponential Decay
    printf("=== Exponential Decay Test ===\n");
    double t0 = 0.0;
    double t_end = 2.0;
    double y0_exp[1] = {1.0};
    double h = 0.01;
    double exact_exp[1] = {exp(-t_end)};
    
    ComparisonResults results_exp;
    memset(&results_exp, 0, sizeof(ComparisonResults));
    
    if (compare_methods(test_exponential, t0, t_end, y0_exp, 1, h, NULL, exact_exp, &results_exp) == 0) {
        print_comparison_results(&results_exp);
        export_comparison_csv("exponential_comparison.csv", &results_exp);
        printf("✅ Exponential decay benchmark complete\n\n");
    } else {
        printf("❌ Exponential decay benchmark failed\n\n");
    }
    
    // Test 2: Harmonic Oscillator
    printf("=== Harmonic Oscillator Test ===\n");
    double y0_osc[2] = {1.0, 0.0};
    double exact_osc[2] = {cos(t_end), -sin(t_end)};
    
    ComparisonResults results_osc;
    memset(&results_osc, 0, sizeof(ComparisonResults));
    
    if (compare_methods(test_oscillator, t0, t_end, y0_osc, 2, h, NULL, exact_osc, &results_osc) == 0) {
        print_comparison_results(&results_osc);
        export_comparison_csv("oscillator_comparison.csv", &results_osc);
        printf("✅ Harmonic oscillator benchmark complete\n\n");
    } else {
        printf("❌ Harmonic oscillator benchmark failed\n\n");
    }
    
    printf("=== All Benchmarks Complete ===\n");
    printf("CSV files exported: exponential_comparison.csv, oscillator_comparison.csv\n");
    
    return 0;
}
