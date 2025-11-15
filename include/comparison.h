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
    // Standard methods
    double euler_time;
    double ddeuler_time;
    double rk3_time;
    double ddrk3_time;
    double am_time;
    double ddam_time;
    
    // Parallel methods
    double parallel_rk3_time;
    double parallel_am_time;
    double parallel_euler_time;
    double stacked_rk3_time;
    double stacked_am_time;
    double stacked_euler_time;
    
    // Errors
    double euler_error;
    double ddeuler_error;
    double rk3_error;
    double ddrk3_error;
    double am_error;
    double ddam_error;
    double parallel_rk3_error;
    double parallel_am_error;
    double parallel_euler_error;
    double stacked_rk3_error;
    double stacked_am_error;
    double stacked_euler_error;
    
    // Accuracies
    double euler_accuracy;
    double ddeuler_accuracy;
    double rk3_accuracy;
    double ddrk3_accuracy;
    double am_accuracy;
    double ddam_accuracy;
    double parallel_rk3_accuracy;
    double parallel_am_accuracy;
    double parallel_euler_accuracy;
    double stacked_rk3_accuracy;
    double stacked_am_accuracy;
    double stacked_euler_accuracy;
    
    // Steps
    size_t euler_steps;
    size_t ddeuler_steps;
    size_t rk3_steps;
    size_t ddrk3_steps;
    size_t am_steps;
    size_t ddam_steps;
    size_t parallel_rk3_steps;
    size_t parallel_am_steps;
    size_t parallel_euler_steps;
    size_t stacked_rk3_steps;
    size_t stacked_am_steps;
    size_t stacked_euler_steps;
    
    // Real-time methods
    double realtime_rk3_time;
    double realtime_am_time;
    double realtime_euler_time;
    double realtime_rk3_error;
    double realtime_am_error;
    double realtime_euler_error;
    double realtime_rk3_accuracy;
    double realtime_am_accuracy;
    double realtime_euler_accuracy;
    
    // Online methods
    double online_rk3_time;
    double online_am_time;
    double online_euler_time;
    double online_rk3_error;
    double online_am_error;
    double online_euler_error;
    double online_rk3_accuracy;
    double online_am_accuracy;
    double online_euler_accuracy;
    
    // Dynamic methods
    double dynamic_rk3_time;
    double dynamic_am_time;
    double dynamic_euler_time;
    double dynamic_rk3_error;
    double dynamic_am_error;
    double dynamic_euler_error;
    double dynamic_rk3_accuracy;
    double dynamic_am_accuracy;
    double dynamic_euler_accuracy;
    
    // Nonlinear programming solvers
    double nonlinear_ode_time;
    double nonlinear_pde_time;
    double nonlinear_ode_error;
    double nonlinear_pde_error;
    double nonlinear_ode_accuracy;
    double nonlinear_pde_accuracy;
    
    // Additional distributed/data-driven/online/real-time solvers
    double distributed_datadriven_time;
    double online_datadriven_time;
    double realtime_datadriven_time;
    double distributed_online_time;
    double distributed_realtime_time;
    
    // Parallel performance metrics
    double speedup_rk3;      // Speedup factor for parallel RK3
    double speedup_am;        // Speedup factor for parallel AM
    double speedup_euler;    // Speedup factor for parallel Euler
    size_t num_workers;       // Number of parallel workers used
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
