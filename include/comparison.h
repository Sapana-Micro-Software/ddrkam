/*
 * Method Comparison Framework
 * RK3 vs DDRK3 vs AM vs DDAM
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef COMPARISON_H
#define COMPARISON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Comparison results structure
 */
typedef struct {
    double euler_time;
    double ddeuler_time;
    double rk3_time;
    double ddrk3_time;
    double am_time;
    double ddam_time;
    
    double euler_error;
    double ddeuler_error;
    double rk3_error;
    double ddrk3_error;
    double am_error;
    double ddam_error;
    
    double euler_accuracy;
    double ddeuler_accuracy;
    double rk3_accuracy;
    double ddrk3_accuracy;
    double am_accuracy;
    double ddam_accuracy;
    
    size_t euler_steps;
    size_t ddeuler_steps;
    size_t rk3_steps;
    size_t ddrk3_steps;
    size_t am_steps;
    size_t ddam_steps;
} ComparisonResults;

/**
 * Run comprehensive comparison of all methods
 * 
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param n: System dimension
 * @param h: Step size
 * @param params: ODE parameters
 * @param exact_solution: Exact solution at t_end (for error calculation)
 * @param results: Output comparison results
 * @return: 0 on success, -1 on failure
 */
int compare_methods(ODEFunction f, double t0, double t_end, const double* y0,
                   size_t n, double h, void* params, const double* exact_solution,
                   ComparisonResults* results);

/**
 * Print comparison results
 */
void print_comparison_results(const ComparisonResults* results);

/**
 * Export comparison results to CSV
 * 
 * @param filename: Output CSV filename
 * @param results: Comparison results
 * @return: 0 on success, -1 on failure
 */
int export_comparison_csv(const char* filename, const ComparisonResults* results);

#ifdef __cplusplus
}
#endif

#endif /* COMPARISON_H */
