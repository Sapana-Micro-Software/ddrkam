/*
 * Comprehensive Benchmark Test Suite
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "../include/rk3.h"
#include "../include/hierarchical_rk.h"
#include "../include/adams.h"
#include "../include/comparison.h"

// Benchmark test functions
void benchmark_exponential(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = -y[0];
}

void benchmark_oscillator(double t, const double* y, double* dydt, void* params) {
    (void)t;
    (void)params;
    dydt[0] = y[1];
    dydt[1] = -y[0];
}

void benchmark_lorenz(double t, const double* y, double* dydt, void* params) {
    (void)t;
    double* p = (double*)params;
    double sigma = p[0], rho = p[1], beta = p[2];
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

typedef struct {
    double time;
    double error;
    double accuracy;
    size_t steps;
    int success;
} BenchmarkResult;

BenchmarkResult benchmark_rk3(ODEFunction f, double t0, double t_end, 
                              const double* y0, size_t n, double h, 
                              void* params, const double* exact) {
    BenchmarkResult result = {0};
    clock_t start = clock();
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 10;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * n * sizeof(double));
    
    if (!t_out || !y_out) {
        return result;
    }
    
    size_t steps = rk3_solve(f, t0, t_end, y0, n, h, params, t_out, y_out);
    clock_t end = clock();
    
    result.time = ((double)(end - start)) / CLOCKS_PER_SEC;
    result.steps = steps;
    
    if (steps > 0 && exact) {
        double error = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = y_out[(steps-1)*n + i] - exact[i];
            error += diff * diff;
        }
        result.error = sqrt(error);
        
        double total_exact = 0.0, total_computed = 0.0;
        for (size_t i = 0; i < n; i++) {
            total_exact += fabs(exact[i]);
            total_computed += fabs(y_out[(steps-1)*n + i]);
        }
        if (total_exact > 0) {
            result.accuracy = 1.0 - fabs(total_computed - total_exact) / total_exact;
        } else {
            result.accuracy = 1.0;
        }
    }
    
    result.success = 1;
    free(t_out);
    free(y_out);
    return result;
}

BenchmarkResult benchmark_ddrk3(ODEFunction f, double t0, double t_end,
                                const double* y0, size_t n, double h,
                                void* params, const double* exact) {
    BenchmarkResult result = {0};
    
    HierarchicalRKSolver solver;
    if (hierarchical_rk_init(&solver, 3, n, 16) != 0) {
        return result;
    }
    
    clock_t start = clock();
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 10;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * n * sizeof(double));
    
    if (!t_out || !y_out) {
        hierarchical_rk_free(&solver);
        return result;
    }
    
    size_t steps = hierarchical_rk_solve(&solver, f, t0, t_end, y0, h, params, t_out, y_out);
    clock_t end = clock();
    
    result.time = ((double)(end - start)) / CLOCKS_PER_SEC;
    result.steps = steps;
    
    if (steps > 0 && exact) {
        double error = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = y_out[(steps-1)*n + i] - exact[i];
            error += diff * diff;
        }
        result.error = sqrt(error);
        
        double total_exact = 0.0, total_computed = 0.0;
        for (size_t i = 0; i < n; i++) {
            total_exact += fabs(exact[i]);
            total_computed += fabs(y_out[(steps-1)*n + i]);
        }
        if (total_exact > 0) {
            result.accuracy = 1.0 - fabs(total_computed - total_exact) / total_exact;
        } else {
            result.accuracy = 1.0;
        }
    }
    
    result.success = 1;
    hierarchical_rk_free(&solver);
    free(t_out);
    free(y_out);
    return result;
}

void run_benchmark_suite() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          DDRKAM Comprehensive Benchmark Suite                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Test 1: Exponential Decay
    printf("=== Test 1: Exponential Decay ===\n");
    double t0 = 0.0, t_end = 2.0;
    double y0_exp[1] = {1.0};
    double h = 0.01;
    double exact_exp[1] = {exp(-t_end)};
    
    BenchmarkResult rk3_exp = benchmark_rk3(benchmark_exponential, t0, t_end, 
                                           y0_exp, 1, h, NULL, exact_exp);
    BenchmarkResult ddrk3_exp = benchmark_ddrk3(benchmark_exponential, t0, t_end,
                                                y0_exp, 1, h, NULL, exact_exp);
    
    printf("RK3:   time=%.6fs, error=%.6e, accuracy=%.6f%%, steps=%zu\n",
           rk3_exp.time, rk3_exp.error, rk3_exp.accuracy * 100, rk3_exp.steps);
    printf("DDRK3: time=%.6fs, error=%.6e, accuracy=%.6f%%, steps=%zu\n",
           ddrk3_exp.time, ddrk3_exp.error, ddrk3_exp.accuracy * 100, ddrk3_exp.steps);
    printf("\n");
    
    // Test 2: Harmonic Oscillator
    printf("=== Test 2: Harmonic Oscillator ===\n");
    t_end = 2.0 * M_PI;
    double y0_osc[2] = {1.0, 0.0};
    double exact_osc[2] = {cos(t_end), -sin(t_end)};
    
    BenchmarkResult rk3_osc = benchmark_rk3(benchmark_oscillator, t0, t_end,
                                            y0_osc, 2, h, NULL, exact_osc);
    BenchmarkResult ddrk3_osc = benchmark_ddrk3(benchmark_oscillator, t0, t_end,
                                                y0_osc, 2, h, NULL, exact_osc);
    
    printf("RK3:   time=%.6fs, error=%.6e, accuracy=%.6f%%, steps=%zu\n",
           rk3_osc.time, rk3_osc.error, rk3_osc.accuracy * 100, rk3_osc.steps);
    printf("DDRK3: time=%.6fs, error=%.6e, accuracy=%.6f%%, steps=%zu\n",
           ddrk3_osc.time, ddrk3_osc.error, ddrk3_osc.accuracy * 100, ddrk3_osc.steps);
    printf("\n");
    
    // Test 3: Lorenz System
    printf("=== Test 3: Lorenz System ===\n");
    t_end = 1.0;
    double params_lorenz[3] = {10.0, 28.0, 8.0/3.0};
    double y0_lorenz[3] = {1.0, 1.0, 1.0};
    // No exact solution for Lorenz, just measure performance
    double exact_lorenz[3] = {0.0, 0.0, 0.0}; // Placeholder
    
    BenchmarkResult rk3_lorenz = benchmark_rk3(benchmark_lorenz, t0, t_end,
                                              y0_lorenz, 3, h, params_lorenz, NULL);
    BenchmarkResult ddrk3_lorenz = benchmark_ddrk3(benchmark_lorenz, t0, t_end,
                                                   y0_lorenz, 3, h, params_lorenz, NULL);
    
    printf("RK3:   time=%.6fs, steps=%zu\n",
           rk3_lorenz.time, rk3_lorenz.steps);
    printf("DDRK3: time=%.6fs, steps=%zu\n",
           ddrk3_lorenz.time, ddrk3_lorenz.steps);
    printf("\n");
    
    // Export benchmark data
    FILE* fp = fopen("benchmark_results.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"exponential\": {\n");
        fprintf(fp, "    \"rk3\": {\"time\": %.6f, \"error\": %.6e, \"accuracy\": %.6f, \"steps\": %zu},\n",
                rk3_exp.time, rk3_exp.error, rk3_exp.accuracy, rk3_exp.steps);
        fprintf(fp, "    \"ddrk3\": {\"time\": %.6f, \"error\": %.6e, \"accuracy\": %.6f, \"steps\": %zu}\n",
                ddrk3_exp.time, ddrk3_exp.error, ddrk3_exp.accuracy, ddrk3_exp.steps);
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"oscillator\": {\n");
        fprintf(fp, "    \"rk3\": {\"time\": %.6f, \"error\": %.6e, \"accuracy\": %.6f, \"steps\": %zu},\n",
                rk3_osc.time, rk3_osc.error, rk3_osc.accuracy, rk3_osc.steps);
        fprintf(fp, "    \"ddrk3\": {\"time\": %.6f, \"error\": %.6e, \"accuracy\": %.6f, \"steps\": %zu}\n",
                ddrk3_osc.time, ddrk3_osc.error, ddrk3_osc.accuracy, ddrk3_osc.steps);
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"lorenz\": {\n");
        fprintf(fp, "    \"rk3\": {\"time\": %.6f, \"steps\": %zu},\n",
                rk3_lorenz.time, rk3_lorenz.steps);
        fprintf(fp, "    \"ddrk3\": {\"time\": %.6f, \"steps\": %zu}\n",
                ddrk3_lorenz.time, ddrk3_lorenz.steps);
        fprintf(fp, "  }\n");
        fprintf(fp, "}\n");
        fclose(fp);
        printf("✅ Benchmark results exported to benchmark_results.json\n");
    }
    
    printf("\n=== Benchmark Summary ===\n");
    printf("All benchmarks completed successfully.\n");
}

int main() {
    run_benchmark_suite();
    return 0;
}
