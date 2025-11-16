/*
 * Method Comparison Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "comparison.h"
#include "euler.h"
#include "hierarchical_euler.h"
#include "rk3.h"
#include "additional_methods.h"
#include "hierarchical_rk.h"
#include "adams.h"
#include "parallel_rk.h"
#include "parallel_adams.h"
#include "parallel_euler.h"
#include "realtime_online.h"
#include "nonlinear_solver.h"
#include "distributed_solvers.h"
#include "optimization_solvers.h"
#include "mapreduce_solvers.h"
#include "spark_solvers.h"
#include "nonorthodox_architectures.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Data-Driven Adams Method - Full implementation combining Adams with hierarchical features
static void ddam_step(ODEFunction f, const double* t, const double* y,
                      size_t n, double h, void* params, double* y_next) {
    // Initialize hierarchical solver for data-driven correction
    HierarchicalRKSolver hierarchical_solver;
    if (hierarchical_rk_init(&hierarchical_solver, 2, n, 16) != 0) {
        // Fallback to standard Adams if hierarchical init fails
        double* y_pred = (double*)malloc(n * sizeof(double));
        double* y_corr = (double*)malloc(n * sizeof(double));
        if (y_pred && y_corr) {
            adams_bashforth3(f, t, y, n, h, params, y_pred);
            adams_moulton3(f, t, y, n, h, params, y_pred, y_corr);
            memcpy(y_next, y_corr, n * sizeof(double));
        }
        if (y_pred) free(y_pred);
        if (y_corr) free(y_corr);
        return;
    }
    
    // Step 1: Adams-Bashforth predictor
    double* y_pred = (double*)malloc(n * sizeof(double));
    if (!y_pred) {
        hierarchical_rk_free(&hierarchical_solver);
        return;
    }
    adams_bashforth3(f, t, y, n, h, params, y_pred);
    
    // Step 2: Adams-Moulton corrector
    double* y_corr = (double*)malloc(n * sizeof(double));
    if (!y_corr) {
        free(y_pred);
        hierarchical_rk_free(&hierarchical_solver);
        return;
    }
    adams_moulton3(f, t, y, n, h, params, y_pred, y_corr);
    
    // Step 3: Apply hierarchical data-driven correction
    // Use hierarchical RK to refine the Adams-Moulton result
    double* y_refined = (double*)malloc(n * sizeof(double));
    if (y_refined) {
        memcpy(y_refined, y_corr, n * sizeof(double));
        // Apply hierarchical correction with smaller step for refinement
        hierarchical_rk_step(&hierarchical_solver, f, t[2] + h, y_refined, h * 0.1, params);
        
        // Blend Adams-Moulton result with hierarchical refinement
        // Weight: 70% Adams-Moulton, 30% hierarchical correction
        for (size_t i = 0; i < n; i++) {
            y_next[i] = 0.7 * y_corr[i] + 0.3 * y_refined[i];
        }
        free(y_refined);
    } else {
        // Fallback to Adams-Moulton if refinement fails
        memcpy(y_next, y_corr, n * sizeof(double));
    }
    
    free(y_pred);
    free(y_corr);
    hierarchical_rk_free(&hierarchical_solver);
}

static double compute_error(const double* computed, const double* exact, size_t n) {
    double error = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = computed[i] - exact[i];
        error += diff * diff;
    }
    return sqrt(error);
}

static double compute_accuracy(const double* computed, const double* exact, size_t n) {
    double total_exact = 0.0;
    double total_computed = 0.0;
    
    for (size_t i = 0; i < n; i++) {
        total_exact += fabs(exact[i]);
        total_computed += fabs(computed[i]);
    }
    
    if (total_exact == 0.0) return 1.0;
    
    return 1.0 - fabs(total_computed - total_exact) / total_exact;
}

int compare_methods(ODEFunction f, double t0, double t_end, const double* y0,
                   size_t n, double h, void* params, const double* exact_solution,
                   ComparisonResults* results) {
    if (!f || !y0 || !exact_solution || !results || n == 0) {
        return -1;
    }
    
    memset(results, 0, sizeof(ComparisonResults));
    
    size_t max_steps = (size_t)((t_end - t0) / h) + 10;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * n * sizeof(double));
    
    if (!t_out || !y_out) {
        if (t_out) free(t_out);
        if (y_out) free(y_out);
        return -1;
    }
    
    clock_t start, end;
    
    // Test Euler's Method
    double* y0_copy = (double*)malloc(n * sizeof(double));
    memcpy(y0_copy, y0, n * sizeof(double));
    
    start = clock();
    size_t euler_steps = euler_solve(f, t0, t_end, y0_copy, n, h, params, t_out, y_out);
    end = clock();
    
    results->euler_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    results->euler_steps = euler_steps;
    
    if (euler_steps > 0) {
        double* final_euler = &y_out[(euler_steps - 1) * n];
        results->euler_error = compute_error(final_euler, exact_solution, n);
        results->euler_accuracy = compute_accuracy(final_euler, exact_solution, n);
    }
    
    free(y0_copy);
    
    // Test DDEuler (Data-Driven Euler)
    HierarchicalEulerSolver ddeuler_solver;
    if (hierarchical_euler_init(&ddeuler_solver, 3, n, 16) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        size_t ddeuler_steps = hierarchical_euler_solve(&ddeuler_solver, f, t0, t_end, y0_copy,
                                                       h, params, t_out, y_out);
        end = clock();
        
        results->ddeuler_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        results->ddeuler_steps = ddeuler_steps;
        
        if (ddeuler_steps > 0) {
            double* final_ddeuler = &y_out[(ddeuler_steps - 1) * n];
            results->ddeuler_error = compute_error(final_ddeuler, exact_solution, n);
            results->ddeuler_accuracy = compute_accuracy(final_ddeuler, exact_solution, n);
        }
        
        hierarchical_euler_free(&ddeuler_solver);
        free(y0_copy);
    }
    
    // Test RK3
    y0_copy = (double*)malloc(n * sizeof(double));
    memcpy(y0_copy, y0, n * sizeof(double));
    
    start = clock();
    size_t rk3_steps = rk3_solve(f, t0, t_end, y0_copy, n, h, params, t_out, y_out);
    end = clock();
    
    results->rk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    results->rk3_steps = rk3_steps;
    
    if (rk3_steps > 0) {
        double* final_rk3 = &y_out[(rk3_steps - 1) * n];
        results->rk3_error = compute_error(final_rk3, exact_solution, n);
        results->rk3_accuracy = compute_accuracy(final_rk3, exact_solution, n);
    }
    
    free(y0_copy);
    
    // Test RK4
    y0_copy = (double*)malloc(n * sizeof(double));
    memcpy(y0_copy, y0, n * sizeof(double));
    
    start = clock();
    size_t rk4_steps = rk4_solve(f, t0, t_end, y0_copy, n, h, params, t_out, y_out);
    end = clock();
    
    results->rk4_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    results->rk4_steps = rk4_steps;
    
    if (rk4_steps > 0) {
        double* final_rk4 = &y_out[(rk4_steps - 1) * n];
        results->rk4_error = compute_error(final_rk4, exact_solution, n);
        results->rk4_accuracy = compute_accuracy(final_rk4, exact_solution, n);
    }
    
    free(y0_copy);
    
    // Test DDRK3 (Hierarchical RK)
    HierarchicalRKSolver ddrk3_solver;
    if (hierarchical_rk_init(&ddrk3_solver, 3, n, 16) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        size_t ddrk3_steps = hierarchical_rk_solve(&ddrk3_solver, f, t0, t_end, y0_copy,
                                                   h, params, t_out, y_out);
        end = clock();
        
        results->ddrk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        results->ddrk3_steps = ddrk3_steps;
        
        if (ddrk3_steps > 0) {
            double* final_ddrk3 = &y_out[(ddrk3_steps - 1) * n];
            results->ddrk3_error = compute_error(final_ddrk3, exact_solution, n);
            results->ddrk3_accuracy = compute_accuracy(final_ddrk3, exact_solution, n);
        }
        
        hierarchical_rk_free(&ddrk3_solver);
        free(y0_copy);
    }
    
    // Test AM (Adams Methods) - simplified implementation
    // Need at least 3 previous points for Adams-Bashforth 3
    if (max_steps >= 3) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        // Initialize with RK3 for first few steps
        double* t_am = (double*)malloc(3 * sizeof(double));
        double* y_am = (double*)malloc(3 * n * sizeof(double));
        
        if (t_am && y_am) {
            // Get first 3 points using RK3
            size_t init_steps = rk3_solve(f, t0, t0 + 2*h, y0_copy, n, h, params, t_am, y_am);
            
            if (init_steps >= 3) {
                start = clock();
                
                double t_current = t_am[2];
                size_t am_steps = 3;
                
                while (t_current < t_end && am_steps < max_steps) {
                    double* y_pred = (double*)malloc(n * sizeof(double));
                    double* y_corr = (double*)malloc(n * sizeof(double));
                    
                    if (y_pred && y_corr) {
                        adams_bashforth3(f, t_am, y_am, n, h, params, y_pred);
                        adams_moulton3(f, t_am, y_am, n, h, params, y_pred, y_corr);
                        
                        // Shift arrays
                        memmove(&t_am[0], &t_am[1], 2 * sizeof(double));
                        memmove(&y_am[0], &y_am[n], 2 * n * sizeof(double));
                        
                        t_am[2] = t_current + h;
                        memcpy(&y_am[2 * n], y_corr, n * sizeof(double));
                        
                        t_current += h;
                        am_steps++;
                    }
                    
                    if (y_pred) free(y_pred);
                    if (y_corr) free(y_corr);
                }
                
                end = clock();
                
                results->am_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->am_steps = am_steps;
                
                if (am_steps > 0) {
                    double* final_am = &y_am[2 * n];
                    results->am_error = compute_error(final_am, exact_solution, n);
                    results->am_accuracy = compute_accuracy(final_am, exact_solution, n);
                }
            }
            
            free(t_am);
            free(y_am);
        }
        
        free(y0_copy);
    }
    
    // Test AM1 (Adams 1st Order - Euler/Implicit Euler)
    if (max_steps >= 1) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        double* t_am1 = (double*)malloc(1 * sizeof(double));
        double* y_am1 = (double*)malloc(1 * n * sizeof(double));
        
        if (t_am1 && y_am1) {
            t_am1[0] = t0;
            memcpy(&y_am1[0 * n], y0_copy, n * sizeof(double));
            
            start = clock();
            
            double t_current = t0;
            size_t am1_steps = 1;
            
            while (t_current < t_end && am1_steps < max_steps) {
                double* y_pred = (double*)malloc(n * sizeof(double));
                double* y_corr = (double*)malloc(n * sizeof(double));
                
                if (y_pred && y_corr) {
                    adams_bashforth1(f, t_am1, y_am1, n, h, params, y_pred);
                    adams_moulton1(f, t_am1, y_am1, n, h, params, y_pred, y_corr);
                    
                    t_am1[0] = t_current + h;
                    memcpy(&y_am1[0 * n], y_corr, n * sizeof(double));
                    
                    t_current += h;
                    am1_steps++;
                }
                
                if (y_pred) free(y_pred);
                if (y_corr) free(y_corr);
            }
            
            end = clock();
            
            results->am1_time = ((double)(end - start)) / CLOCKS_PER_SEC;
            results->am1_steps = am1_steps;
            
            if (am1_steps > 0) {
                double* final_am1 = &y_am1[0 * n];
                results->am1_error = compute_error(final_am1, exact_solution, n);
                results->am1_accuracy = compute_accuracy(final_am1, exact_solution, n);
            }
            
            free(t_am1);
            free(y_am1);
        }
        
        free(y0_copy);
    }
    
    // Test AM2 (Adams 2nd Order)
    if (max_steps >= 2) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        double* t_am2 = (double*)malloc(2 * sizeof(double));
        double* y_am2 = (double*)malloc(2 * n * sizeof(double));
        
        if (t_am2 && y_am2) {
            // Get first 2 points using RK3
            size_t init_steps = rk3_solve(f, t0, t0 + h, y0_copy, n, h, params, t_am2, y_am2);
            
            if (init_steps >= 2) {
                start = clock();
                
                double t_current = t_am2[1];
                size_t am2_steps = 2;
                
                while (t_current < t_end && am2_steps < max_steps) {
                    double* y_pred = (double*)malloc(n * sizeof(double));
                    double* y_corr = (double*)malloc(n * sizeof(double));
                    
                    if (y_pred && y_corr) {
                        adams_bashforth2(f, t_am2, y_am2, n, h, params, y_pred);
                        adams_moulton2(f, t_am2, y_am2, n, h, params, y_pred, y_corr);
                        
                        // Shift arrays
                        memmove(&t_am2[0], &t_am2[1], 1 * sizeof(double));
                        memmove(&y_am2[0], &y_am2[n], 1 * n * sizeof(double));
                        
                        t_am2[1] = t_current + h;
                        memcpy(&y_am2[1 * n], y_corr, n * sizeof(double));
                        
                        t_current += h;
                        am2_steps++;
                    }
                    
                    if (y_pred) free(y_pred);
                    if (y_corr) free(y_corr);
                }
                
                end = clock();
                
                results->am2_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->am2_steps = am2_steps;
                
                if (am2_steps > 0) {
                    double* final_am2 = &y_am2[1 * n];
                    results->am2_error = compute_error(final_am2, exact_solution, n);
                    results->am2_accuracy = compute_accuracy(final_am2, exact_solution, n);
                }
            }
            
            free(t_am2);
            free(y_am2);
        }
        
        free(y0_copy);
    }
    
    // Test AM4 (Adams 4th Order)
    if (max_steps >= 4) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        double* t_am4 = (double*)malloc(4 * sizeof(double));
        double* y_am4 = (double*)malloc(4 * n * sizeof(double));
        
        if (t_am4 && y_am4) {
            // Get first 4 points using RK3
            size_t init_steps = rk3_solve(f, t0, t0 + 3*h, y0_copy, n, h, params, t_am4, y_am4);
            
            if (init_steps >= 4) {
                start = clock();
                
                double t_current = t_am4[3];
                size_t am4_steps = 4;
                
                while (t_current < t_end && am4_steps < max_steps) {
                    double* y_pred = (double*)malloc(n * sizeof(double));
                    double* y_corr = (double*)malloc(n * sizeof(double));
                    
                    if (y_pred && y_corr) {
                        adams_bashforth4(f, t_am4, y_am4, n, h, params, y_pred);
                        adams_moulton4(f, t_am4, y_am4, n, h, params, y_pred, y_corr);
                        
                        // Shift arrays
                        memmove(&t_am4[0], &t_am4[1], 3 * sizeof(double));
                        memmove(&y_am4[0], &y_am4[n], 3 * n * sizeof(double));
                        
                        t_am4[3] = t_current + h;
                        memcpy(&y_am4[3 * n], y_corr, n * sizeof(double));
                        
                        t_current += h;
                        am4_steps++;
                    }
                    
                    if (y_pred) free(y_pred);
                    if (y_corr) free(y_corr);
                }
                
                end = clock();
                
                results->am4_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->am4_steps = am4_steps;
                
                if (am4_steps > 0) {
                    double* final_am4 = &y_am4[3 * n];
                    results->am4_error = compute_error(final_am4, exact_solution, n);
                    results->am4_accuracy = compute_accuracy(final_am4, exact_solution, n);
                }
            }
            
            free(t_am4);
            free(y_am4);
        }
        
        free(y0_copy);
    }
    
    // Test AM5 (Adams 5th Order)
    if (max_steps >= 5) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        double* t_am5 = (double*)malloc(5 * sizeof(double));
        double* y_am5 = (double*)malloc(5 * n * sizeof(double));
        
        if (t_am5 && y_am5) {
            // Get first 5 points using RK3
            size_t init_steps = rk3_solve(f, t0, t0 + 4*h, y0_copy, n, h, params, t_am5, y_am5);
            
            if (init_steps >= 5) {
                start = clock();
                
                double t_current = t_am5[4];
                size_t am5_steps = 5;
                
                while (t_current < t_end && am5_steps < max_steps) {
                    double* y_pred = (double*)malloc(n * sizeof(double));
                    double* y_corr = (double*)malloc(n * sizeof(double));
                    
                    if (y_pred && y_corr) {
                        adams_bashforth5(f, t_am5, y_am5, n, h, params, y_pred);
                        adams_moulton5(f, t_am5, y_am5, n, h, params, y_pred, y_corr);
                        
                        // Shift arrays
                        memmove(&t_am5[0], &t_am5[1], 4 * sizeof(double));
                        memmove(&y_am5[0], &y_am5[n], 4 * n * sizeof(double));
                        
                        t_am5[4] = t_current + h;
                        memcpy(&y_am5[4 * n], y_corr, n * sizeof(double));
                        
                        t_current += h;
                        am5_steps++;
                    }
                    
                    if (y_pred) free(y_pred);
                    if (y_corr) free(y_corr);
                }
                
                end = clock();
                
                results->am5_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->am5_steps = am5_steps;
                
                if (am5_steps > 0) {
                    double* final_am5 = &y_am5[4 * n];
                    results->am5_error = compute_error(final_am5, exact_solution, n);
                    results->am5_accuracy = compute_accuracy(final_am5, exact_solution, n);
                }
            }
            
            free(t_am5);
            free(y_am5);
        }
        
        free(y0_copy);
    }
    
    // Test DDAM (Data-Driven Adams)
    HierarchicalRKSolver ddam_solver;
    if (hierarchical_rk_init(&ddam_solver, 2, n, 16) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        // Initialize with hierarchical RK
        double* t_ddam = (double*)malloc(3 * sizeof(double));
        double* y_ddam = (double*)malloc(3 * n * sizeof(double));
        
        if (t_ddam && y_ddam) {
            size_t init_steps = hierarchical_rk_solve(&ddam_solver, f, t0, t0 + 2*h,
                                                      y0_copy, h, params, t_ddam, y_ddam);
            
            if (init_steps >= 3) {
                start = clock();
                
                double t_current = t_ddam[2];
                size_t ddam_steps = 3;
                
                while (t_current < t_end && ddam_steps < max_steps) {
                    double* y_next = (double*)malloc(n * sizeof(double));
                    
                    if (y_next) {
                        ddam_step(f, t_ddam, y_ddam, n, h, params, y_next);
                        
                        // Shift arrays
                        memmove(&t_ddam[0], &t_ddam[1], 2 * sizeof(double));
                        memmove(&y_ddam[0], &y_ddam[n], 2 * n * sizeof(double));
                        
                        t_ddam[2] = t_current + h;
                        memcpy(&y_ddam[2 * n], y_next, n * sizeof(double));
                        
                        t_current += h;
                        ddam_steps++;
                    }
                    
                    if (y_next) free(y_next);
                }
                
                end = clock();
                
                results->ddam_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->ddam_steps = ddam_steps;
                
                if (ddam_steps > 0) {
                    double* final_ddam = &y_ddam[2 * n];
                    results->ddam_error = compute_error(final_ddam, exact_solution, n);
                    results->ddam_accuracy = compute_accuracy(final_ddam, exact_solution, n);
                }
            }
            
            free(t_ddam);
            free(y_ddam);
        }
        
        hierarchical_rk_free(&ddam_solver);
        free(y0_copy);
    }
    
    // Test Parallel RK3
    ParallelRKSolver parallel_rk3_solver;
    if (parallel_rk_init(&parallel_rk3_solver, n, 4, PARALLEL_OPENMP, NULL) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        size_t parallel_rk3_steps = parallel_rk_solve(&parallel_rk3_solver, f, t0, t_end, y0_copy,
                                                     h, params, t_out, y_out);
        end = clock();
        
        results->parallel_rk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        results->parallel_rk3_steps = parallel_rk3_steps;
        results->num_workers = 4;
        
        if (parallel_rk3_steps > 0) {
            double* final_parallel_rk3 = &y_out[(parallel_rk3_steps - 1) * n];
            results->parallel_rk3_error = compute_error(final_parallel_rk3, exact_solution, n);
            results->parallel_rk3_accuracy = compute_accuracy(final_parallel_rk3, exact_solution, n);
            if (results->rk3_time > 0) {
                results->speedup_rk3 = results->rk3_time / results->parallel_rk3_time;
            }
        }
        
        parallel_rk_free(&parallel_rk3_solver);
        free(y0_copy);
    }
    
    // Test Stacked RK3
    StackedConfig stacked_config = {
        .num_layers = 3,
        .layer_dims = NULL,
        .hidden_dim = 32,
        .learning_rate = 0.01,
        .use_attention = 1,
        .use_residual = 1
    };
    ParallelRKSolver stacked_rk3_solver;
    if (parallel_rk_init(&stacked_rk3_solver, n, 4, PARALLEL_OPENMP, &stacked_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        size_t stacked_rk3_steps = parallel_rk_solve(&stacked_rk3_solver, f, t0, t_end, y0_copy,
                                                    h, params, t_out, y_out);
        end = clock();
        
        results->stacked_rk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        results->stacked_rk3_steps = stacked_rk3_steps;
        
        if (stacked_rk3_steps > 0) {
            double* final_stacked_rk3 = &y_out[(stacked_rk3_steps - 1) * n];
            results->stacked_rk3_error = compute_error(final_stacked_rk3, exact_solution, n);
            results->stacked_rk3_accuracy = compute_accuracy(final_stacked_rk3, exact_solution, n);
        }
        
        parallel_rk_free(&stacked_rk3_solver);
        free(y0_copy);
    }
    
    // Test Parallel AM
    ParallelAdamsSolver parallel_am_solver;
    if (parallel_adams_init(&parallel_am_solver, n, 4, PARALLEL_OPENMP, NULL) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        size_t parallel_am_steps = parallel_adams_solve(&parallel_am_solver, f, t0, t_end, y0_copy,
                                                      h, params, t_out, y_out);
        end = clock();
        
        results->parallel_am_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        results->parallel_am_steps = parallel_am_steps;
        
        if (parallel_am_steps > 0) {
            double* final_parallel_am = &y_out[(parallel_am_steps - 1) * n];
            results->parallel_am_error = compute_error(final_parallel_am, exact_solution, n);
            results->parallel_am_accuracy = compute_accuracy(final_parallel_am, exact_solution, n);
            if (results->am_time > 0) {
                results->speedup_am = results->am_time / results->parallel_am_time;
            }
        }
        
        parallel_adams_free(&parallel_am_solver);
        free(y0_copy);
    }
    
    // Test Parallel Euler
    ParallelEulerSolver parallel_euler_solver;
    if (parallel_euler_init(&parallel_euler_solver, n, 4, PARALLEL_OPENMP, NULL) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        size_t parallel_euler_steps = parallel_euler_solve(&parallel_euler_solver, f, t0, t_end, y0_copy,
                                                          h, params, t_out, y_out);
        end = clock();
        
        results->parallel_euler_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        results->parallel_euler_steps = parallel_euler_steps;
        
        if (parallel_euler_steps > 0) {
            double* final_parallel_euler = &y_out[(parallel_euler_steps - 1) * n];
            results->parallel_euler_error = compute_error(final_parallel_euler, exact_solution, n);
            results->parallel_euler_accuracy = compute_accuracy(final_parallel_euler, exact_solution, n);
            if (results->euler_time > 0) {
                results->speedup_euler = results->euler_time / results->parallel_euler_time;
            }
        }
        
        parallel_euler_free(&parallel_euler_solver);
        free(y0_copy);
    }
    
    // Test Real-Time RK3
    RealtimeRKSolver realtime_rk3_solver;
    if (realtime_rk_init(&realtime_rk3_solver, n, h, NULL, NULL) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double t_current = t0;
        size_t realtime_steps = 0;
        while (t_current < t_end && realtime_steps < max_steps) {
            double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
            t_current = realtime_rk_step(&realtime_rk3_solver, f, t_current, y0_copy, h_actual, params);
            realtime_steps++;
        }
        end = clock();
        
        results->realtime_rk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        if (realtime_steps > 0) {
            results->realtime_rk3_error = compute_error(y0_copy, exact_solution, n);
            results->realtime_rk3_accuracy = compute_accuracy(y0_copy, exact_solution, n);
        }
        
        realtime_rk_free(&realtime_rk3_solver);
        free(y0_copy);
    }
    
    // Test Online RK3
    OnlineRKSolver online_rk3_solver;
    if (online_rk_init(&online_rk3_solver, n, h, 0.01) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double t_current = t0;
        size_t online_steps = 0;
        while (t_current < t_end && online_steps < max_steps) {
            t_current = online_rk_step(&online_rk3_solver, f, t_current, y0_copy, params);
            online_steps++;
        }
        end = clock();
        
        results->online_rk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        if (online_steps > 0) {
            results->online_rk3_error = compute_error(y0_copy, exact_solution, n);
            results->online_rk3_accuracy = compute_accuracy(y0_copy, exact_solution, n);
        }
        
        online_rk_free(&online_rk3_solver);
        free(y0_copy);
    }
    
    // Test Dynamic RK3
    DynamicRKSolver dynamic_rk3_solver;
    if (dynamic_rk_init(&dynamic_rk3_solver, n, h, 0.01) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double t_current = t0;
        size_t dynamic_steps = 0;
        while (t_current < t_end && dynamic_steps < max_steps) {
            t_current = dynamic_rk_step(&dynamic_rk3_solver, f, t_current, y0_copy, params);
            dynamic_steps++;
        }
        end = clock();
        
        results->dynamic_rk3_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        if (dynamic_steps > 0) {
            results->dynamic_rk3_error = compute_error(y0_copy, exact_solution, n);
            results->dynamic_rk3_accuracy = compute_accuracy(y0_copy, exact_solution, n);
        }
        
        dynamic_rk_free(&dynamic_rk3_solver);
        free(y0_copy);
    }
    
    // Test Nonlinear ODE Solver
    NonlinearODESolver nonlinear_ode;
    if (nonlinear_ode_init(&nonlinear_ode, n, NLP_GRADIENT_DESCENT, NULL, NULL, NULL) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_nlp = (double*)malloc(n * sizeof(double));
        if (y_nlp) {
            nonlinear_ode_solve(&nonlinear_ode, f, t0, t_end, y0_copy, y_nlp);
            end = clock();
            
            results->nonlinear_ode_time = ((double)(end - start)) / CLOCKS_PER_SEC;
            results->nonlinear_ode_error = compute_error(y_nlp, exact_solution, n);
            results->nonlinear_ode_accuracy = compute_accuracy(y_nlp, exact_solution, n);
            free(y_nlp);
        }
        
        nonlinear_ode_free(&nonlinear_ode);
        free(y0_copy);
    }
    
    // Test Distributed Data-Driven Solver
    DistributedDataDrivenSolver dist_dd;
    if (distributed_datadriven_init(&dist_dd, n, 4, 3) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_ddd = (double*)malloc(n * sizeof(double));
        if (y_ddd) {
            distributed_datadriven_solve(&dist_dd, f, t0, t_end, y0_copy, h, params, y_ddd);
            end = clock();
            
            results->distributed_datadriven_time = ((double)(end - start)) / CLOCKS_PER_SEC;
            free(y_ddd);
        }
        
        distributed_datadriven_free(&dist_dd);
        free(y0_copy);
    }
    
    // Test Online Data-Driven Solver
    OnlineDataDrivenSolver online_dd;
    if (online_datadriven_init(&online_dd, n, 3, h, 0.01) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_odd = (double*)malloc(n * sizeof(double));
        if (y_odd) {
            online_datadriven_solve(&online_dd, f, t0, t_end, y0_copy, params, y_odd);
            end = clock();
            
            results->online_datadriven_time = ((double)(end - start)) / CLOCKS_PER_SEC;
            free(y_odd);
        }
        
        online_datadriven_free(&online_dd);
        free(y0_copy);
    }
    
    // Test Karmarkar's Algorithm
    KarmarkarSolver karmarkar;
    double* c = (double*)malloc(n * sizeof(double));
    if (c) {
        for (size_t i = 0; i < n; i++) {
            c[i] = 1.0; // Minimize sum of state variables
        }
        
        if (karmarkar_solver_init(&karmarkar, n, ADAM_ODE, 0.25, 0.5, 1.0, 1e-6,
                                   c, NULL, NULL, 0) == 0) {
            y0_copy = (double*)malloc(n * sizeof(double));
            memcpy(y0_copy, y0, n * sizeof(double));
            
            start = clock();
            double* y_karmarkar = (double*)malloc(n * sizeof(double));
            if (y_karmarkar) {
                if (karmarkar_ode_solve(&karmarkar, f, t0, t_end, y0_copy, params, y_karmarkar) == 0) {
                    end = clock();
                    
                    results->karmarkar_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                    results->karmarkar_steps = (size_t)((t_end - t0) / h);
                    results->karmarkar_iterations = karmarkar.current_iteration;
                    results->karmarkar_error = compute_error(y_karmarkar, exact_solution, n);
                    results->karmarkar_accuracy = compute_accuracy(y_karmarkar, exact_solution, n);
                }
                free(y_karmarkar);
            }
            
            karmarkar_solver_free(&karmarkar);
            free(y0_copy);
        }
        
        free(c);
    }
    
    // Test Map/Reduce Framework
    MapReduceODESolver mapreduce;
    MapReduceConfig mr_config = {
        .num_mappers = 4,
        .num_reducers = 2,
        .chunk_size = n / 4,
        .enable_redundancy = 1,
        .redundancy_factor = 3,
        .use_commodity_hardware = 1,
        .network_bandwidth = 100.0, // 100 MB/s
        .compute_cost_per_hour = 0.10 // $0.10 per hour per node
    };
    
    if (mapreduce_ode_init(&mapreduce, n, &mr_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_mapreduce = (double*)malloc(n * sizeof(double));
        if (y_mapreduce) {
            if (mapreduce_ode_solve(&mapreduce, f, t0, t_end, y0_copy, h, params, y_mapreduce) == 0) {
                end = clock();
                
                results->mapreduce_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->mapreduce_steps = (size_t)((t_end - t0) / h);
                results->mapreduce_map_time = mapreduce.map_time;
                results->mapreduce_reduce_time = mapreduce.reduce_time;
                results->mapreduce_shuffle_time = mapreduce.shuffle_time;
                results->mapreduce_error = compute_error(y_mapreduce, exact_solution, n);
                results->mapreduce_accuracy = compute_accuracy(y_mapreduce, exact_solution, n);
                
                double compute_hours, network_cost;
                results->mapreduce_cost = mapreduce_estimate_cost(&mapreduce, &compute_hours, &network_cost);
            }
            free(y_mapreduce);
        }
        
        mapreduce_ode_free(&mapreduce);
        free(y0_copy);
    }
    
    // Test Spark Framework
    SparkODESolver spark;
    SparkConfig spark_config = {
        .num_executors = 4,
        .cores_per_executor = 2,
        .memory_per_executor = 2048, // 2GB
        .num_partitions = 8,
        .enable_caching = 1,
        .enable_checkpointing = 1,
        .checkpoint_interval = 1.0,
        .use_commodity_hardware = 1,
        .network_bandwidth = 100.0, // 100 MB/s
        .compute_cost_per_hour = 0.10, // $0.10 per hour per executor
        .enable_dynamic_allocation = 1
    };
    
    if (spark_ode_init(&spark, n, &spark_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_spark = (double*)malloc(n * sizeof(double));
        if (y_spark) {
            if (spark_ode_solve(&spark, f, t0, t_end, y0_copy, h, params, y_spark) == 0) {
                end = clock();
                
                results->spark_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->spark_steps = (size_t)((t_end - t0) / h);
                results->spark_map_time = spark.map_time;
                results->spark_reduce_time = spark.reduce_time;
                results->spark_shuffle_time = spark.shuffle_time;
                results->spark_cache_hit_rate = spark.cache_hit_rate;
                results->spark_error = compute_error(y_spark, exact_solution, n);
                results->spark_accuracy = compute_accuracy(y_spark, exact_solution, n);
                
                double compute_hours, network_cost, storage_cost;
                results->spark_cost = spark_estimate_cost(&spark, &compute_hours, &network_cost, &storage_cost);
            }
            free(y_spark);
        }
        
        spark_ode_free(&spark);
        free(y0_copy);
    }
    
    // Test Micro-Gas Jet Circuit
    MicroGasJetSolver microgasjet;
    MicroGasJetConfig mgj_config = {
        .num_jets = n * 2,
        .num_channels = n * 3,
        .flow_rate = 1e-6, // 1 µL/s
        .pressure = 101325.0, // 1 atm
        .temperature = 300.0, // 300 K
        .viscosity = 1.8e-5, // Air viscosity
        .channel_width = 1e-4, // 100 µm
        .channel_length = 1e-3, // 1 mm
        .enable_turbulence = 0,
        .reynolds_number = 100.0
    };
    
    if (microgasjet_ode_init(&microgasjet, n, &mgj_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_mgj = (double*)malloc(n * sizeof(double));
        if (y_mgj) {
            if (microgasjet_ode_solve(&microgasjet, f, t0, t_end, y0_copy, h, params, y_mgj) == 0) {
                end = clock();
                
                results->microgasjet_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->microgasjet_steps = (size_t)((t_end - t0) / h);
                results->microgasjet_flow_energy = microgasjet.total_flow_energy;
                results->microgasjet_error = compute_error(y_mgj, exact_solution, n);
                results->microgasjet_accuracy = compute_accuracy(y_mgj, exact_solution, n);
            }
            free(y_mgj);
        }
        
        microgasjet_ode_free(&microgasjet);
        free(y0_copy);
    }
    
    // Test Dataflow (Arvind)
    DataflowSolver dataflow;
    DataflowConfig df_config = {
        .num_processing_elements = 8,
        .token_buffer_size = 64,
        .instruction_memory_size = 1024,
        .token_matching_time = 1.0, // 1 ns
        .instruction_exec_time = 2.0, // 2 ns
        .enable_tagged_tokens = 1,
        .enable_dynamic_scheduling = 1
    };
    
    if (dataflow_ode_init(&dataflow, n, &df_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_df = (double*)malloc(n * sizeof(double));
        if (y_df) {
            if (dataflow_ode_solve(&dataflow, f, t0, t_end, y0_copy, h, params, y_df) == 0) {
                end = clock();
                
                results->dataflow_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->dataflow_steps = (size_t)((t_end - t0) / h);
                results->dataflow_tokens = dataflow.token_count;
                results->dataflow_token_matching_time = dataflow.token_matching_time;
                results->dataflow_error = compute_error(y_df, exact_solution, n);
                results->dataflow_accuracy = compute_accuracy(y_df, exact_solution, n);
            }
            free(y_df);
        }
        
        dataflow_ode_free(&dataflow);
        free(y0_copy);
    }
    
    // Test ACE (Turing)
    ACESolver ace;
    ACEConfig ace_config = {
        .memory_size = 1024,
        .instruction_width = 32,
        .data_width = 64,
        .clock_frequency = 1e6, // 1 MHz (historical)
        .num_arithmetic_units = 1,
        .enable_pipelining = 0,
        .enable_branch_prediction = 0
    };
    
    if (ace_ode_init(&ace, n, &ace_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_ace = (double*)malloc(n * sizeof(double));
        if (y_ace) {
            if (ace_ode_solve(&ace, f, t0, t_end, y0_copy, h, params, y_ace) == 0) {
                end = clock();
                
                results->ace_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->ace_steps = (size_t)((t_end - t0) / h);
                results->ace_instructions = ace.instructions_executed;
                results->ace_memory_time = ace.memory_access_time;
                results->ace_error = compute_error(y_ace, exact_solution, n);
                results->ace_accuracy = compute_accuracy(y_ace, exact_solution, n);
            }
            free(y_ace);
        }
        
        ace_ode_free(&ace);
        free(y0_copy);
    }
    
    // Test Systolic Array
    SystolicArraySolver systolic;
    SystolicArrayConfig sa_config = {
        .array_rows = 8,
        .array_cols = 8,
        .pe_memory_size = 256,
        .pe_clock_frequency = 1e9, // 1 GHz
        .communication_latency = 1.0, // 1 ns
        .enable_pipelining = 1,
        .topology = 0 // Mesh
    };
    
    if (systolic_ode_init(&systolic, n, &sa_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_sa = (double*)malloc(n * sizeof(double));
        if (y_sa) {
            if (systolic_ode_solve(&systolic, f, t0, t_end, y0_copy, h, params, y_sa) == 0) {
                end = clock();
                
                results->systolic_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->systolic_steps = (size_t)((t_end - t0) / h);
                results->systolic_communication_time = systolic.communication_time;
                results->systolic_error = compute_error(y_sa, exact_solution, n);
                results->systolic_accuracy = compute_accuracy(y_sa, exact_solution, n);
            }
            free(y_sa);
        }
        
        systolic_ode_free(&systolic);
        free(y0_copy);
    }
    
    // Test TPU (Patterson)
    TPUSolver tpu;
    TPUConfig tpu_config = {
        .matrix_unit_size = 128,
        .accumulator_size = 4096,
        .unified_buffer_size = 24, // 24 MB
        .weight_fifo_size = 4, // 4 MB
        .clock_frequency = 700.0, // 700 MHz
        .enable_quantization = 0,
        .precision_bits = 32
    };
    
    if (tpu_ode_init(&tpu, n, &tpu_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_tpu = (double*)malloc(n * sizeof(double));
        if (y_tpu) {
            if (tpu_ode_solve(&tpu, f, t0, t_end, y0_copy, h, params, y_tpu) == 0) {
                end = clock();
                
                results->tpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->tpu_steps = (size_t)((t_end - t0) / h);
                results->tpu_matrix_ops = tpu.matrix_ops;
                results->tpu_bandwidth_utilization = tpu.memory_bandwidth_utilization;
                results->tpu_error = compute_error(y_tpu, exact_solution, n);
                results->tpu_accuracy = compute_accuracy(y_tpu, exact_solution, n);
            }
            free(y_tpu);
        }
        
        tpu_ode_free(&tpu);
        free(y0_copy);
    }
    
    // Test GPU (CUDA)
    GPUSolver gpu_cuda;
    GPUConfig gpu_cuda_config = {
        .gpu_type = ARCH_GPU_CUDA,
        .num_cores = 2560,
        .num_simd_lanes = 32,
        .shared_memory_size = 48, // 48 KB
        .global_memory_size = 8, // 8 GB
        .memory_bandwidth = 900.0, // 900 GB/s
        .warp_size = 32,
        .num_blocks = 256,
        .threads_per_block = 256,
        .enable_tensor_cores = 1
    };
    
    if (gpu_ode_init(&gpu_cuda, n, &gpu_cuda_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_gpu = (double*)malloc(n * sizeof(double));
        if (y_gpu) {
            if (gpu_ode_solve(&gpu_cuda, f, t0, t_end, y0_copy, h, params, y_gpu) == 0) {
                end = clock();
                
                results->gpu_cuda_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->gpu_cuda_steps = (size_t)((t_end - t0) / h);
                results->gpu_cuda_kernel_launches = gpu_cuda.kernel_launches;
                results->gpu_cuda_memory_transfer_time = gpu_cuda.memory_transfer_time;
                results->gpu_cuda_error = compute_error(y_gpu, exact_solution, n);
                results->gpu_cuda_accuracy = compute_accuracy(y_gpu, exact_solution, n);
            }
            free(y_gpu);
        }
        
        gpu_ode_free(&gpu_cuda);
        free(y0_copy);
    }
    
    // Test GPU (Metal)
    GPUSolver gpu_metal;
    GPUConfig gpu_metal_config = {
        .gpu_type = ARCH_GPU_METAL,
        .num_cores = 1024,
        .num_simd_lanes = 32,
        .shared_memory_size = 32, // 32 KB
        .global_memory_size = 16, // 16 GB
        .memory_bandwidth = 400.0, // 400 GB/s
        .warp_size = 32,
        .num_blocks = 128,
        .threads_per_block = 256,
        .enable_tensor_cores = 0
    };
    
    if (gpu_ode_init(&gpu_metal, n, &gpu_metal_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_gpu = (double*)malloc(n * sizeof(double));
        if (y_gpu) {
            if (gpu_ode_solve(&gpu_metal, f, t0, t_end, y0_copy, h, params, y_gpu) == 0) {
                end = clock();
                
                results->gpu_metal_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->gpu_metal_steps = (size_t)((t_end - t0) / h);
                results->gpu_metal_error = compute_error(y_gpu, exact_solution, n);
                results->gpu_metal_accuracy = compute_accuracy(y_gpu, exact_solution, n);
            }
            free(y_gpu);
        }
        
        gpu_ode_free(&gpu_metal);
        free(y0_copy);
    }
    
    // Test GPU (Vulkan)
    GPUSolver gpu_vulkan;
    GPUConfig gpu_vulkan_config = {
        .gpu_type = ARCH_GPU_VULKAN,
        .num_cores = 2048,
        .num_simd_lanes = 32,
        .shared_memory_size = 48, // 48 KB
        .global_memory_size = 12, // 12 GB
        .memory_bandwidth = 600.0, // 600 GB/s
        .warp_size = 32,
        .num_blocks = 256,
        .threads_per_block = 256,
        .enable_tensor_cores = 0
    };
    
    if (gpu_ode_init(&gpu_vulkan, n, &gpu_vulkan_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_gpu = (double*)malloc(n * sizeof(double));
        if (y_gpu) {
            if (gpu_ode_solve(&gpu_vulkan, f, t0, t_end, y0_copy, h, params, y_gpu) == 0) {
                end = clock();
                
                results->gpu_vulkan_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->gpu_vulkan_steps = (size_t)((t_end - t0) / h);
                results->gpu_vulkan_error = compute_error(y_gpu, exact_solution, n);
                results->gpu_vulkan_accuracy = compute_accuracy(y_gpu, exact_solution, n);
            }
            free(y_gpu);
        }
        
        gpu_ode_free(&gpu_vulkan);
        free(y0_copy);
    }
    
    // Test GPU (AMD)
    GPUSolver gpu_amd;
    GPUConfig gpu_amd_config = {
        .gpu_type = ARCH_GPU_AMD,
        .num_cores = 2560,
        .num_simd_lanes = 64, // AMD uses wider SIMD
        .shared_memory_size = 64, // 64 KB
        .global_memory_size = 16, // 16 GB
        .memory_bandwidth = 1000.0, // 1 TB/s
        .warp_size = 64, // Wavefront size
        .num_blocks = 256,
        .threads_per_block = 256,
        .enable_tensor_cores = 0
    };
    
    if (gpu_ode_init(&gpu_amd, n, &gpu_amd_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_gpu = (double*)malloc(n * sizeof(double));
        if (y_gpu) {
            if (gpu_ode_solve(&gpu_amd, f, t0, t_end, y0_copy, h, params, y_gpu) == 0) {
                end = clock();
                
                results->gpu_amd_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->gpu_amd_steps = (size_t)((t_end - t0) / h);
                results->gpu_amd_error = compute_error(y_gpu, exact_solution, n);
                results->gpu_amd_accuracy = compute_accuracy(y_gpu, exact_solution, n);
            }
            free(y_gpu);
        }
        
        gpu_ode_free(&gpu_amd);
        free(y0_copy);
    }
    
    // Test Massively-Threaded (Korf)
    MassivelyThreadedSolver massively_threaded;
    MassivelyThreadedConfig mt_config = {
        .num_threads = 1024,
        .frontier_size = 4096,
        .work_stealing_queue = 2048,
        .thread_spawn_time = 10.0, // 10 ns
        .enable_tail_recursion = 1,
        .enable_work_stealing = 1
    };
    
    if (massively_threaded_ode_init(&massively_threaded, n, &mt_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_mt = (double*)malloc(n * sizeof(double));
        if (y_mt) {
            if (massively_threaded_ode_solve(&massively_threaded, f, t0, t_end, y0_copy, h, params, y_mt) == 0) {
                end = clock();
                
                results->massively_threaded_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->massively_threaded_steps = (size_t)((t_end - t0) / h);
                results->massively_threaded_nodes_expanded = massively_threaded.nodes_expanded;
                results->massively_threaded_error = compute_error(y_mt, exact_solution, n);
                results->massively_threaded_accuracy = compute_accuracy(y_mt, exact_solution, n);
            }
            free(y_mt);
        }
        
        massively_threaded_ode_free(&massively_threaded);
        free(y0_copy);
    }
    
    // Test STARR (Chandra et al.)
    STARRSolver starr;
    STARRConfig starr_config = {
        .num_cores = 64,
        .semantic_memory_size = 1024, // 1 MB
        .associative_memory_size = 512, // 512 KB
        .core_frequency = 2000.0, // 2 GHz
        .enable_semantic_caching = 1,
        .enable_associative_search = 1
    };
    
    if (starr_ode_init(&starr, n, &starr_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_starr = (double*)malloc(n * sizeof(double));
        if (y_starr) {
            if (starr_ode_solve(&starr, f, t0, t_end, y0_copy, h, params, y_starr) == 0) {
                end = clock();
                
                results->starr_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->starr_steps = (size_t)((t_end - t0) / h);
                results->starr_semantic_hits = starr.semantic_hits;
                results->starr_associative_hits = starr.associative_hits;
                results->starr_error = compute_error(y_starr, exact_solution, n);
                results->starr_accuracy = compute_accuracy(y_starr, exact_solution, n);
            }
            free(y_starr);
        }
        
        starr_ode_free(&starr);
        free(y0_copy);
    }
    
    // Test TrueNorth (IBM)
    TrueNorthSolver truenorth;
    TrueNorthConfig tn_config = {
        .num_cores = 4096,
        .neurons_per_core = 256,
        .synapses_per_core = 1024,
        .neuron_firing_rate = 1000.0, // 1 kHz
        .enable_spike_timing = 1,
        .enable_learning = 1
    };
    
    if (truenorth_ode_init(&truenorth, n, &tn_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_tn = (double*)malloc(n * sizeof(double));
        if (y_tn) {
            if (truenorth_ode_solve(&truenorth, f, t0, t_end, y0_copy, h, params, y_tn) == 0) {
                end = clock();
                
                results->truenorth_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->truenorth_steps = (size_t)((t_end - t0) / h);
                results->truenorth_spikes = truenorth.total_spikes;
                results->truenorth_energy = truenorth.energy_consumption;
                results->truenorth_error = compute_error(y_tn, exact_solution, n);
                results->truenorth_accuracy = compute_accuracy(y_tn, exact_solution, n);
            }
            free(y_tn);
        }
        
        truenorth_ode_free(&truenorth);
        free(y0_copy);
    }
    
    // Test Loihi (Intel)
    LoihiSolver loihi;
    LoihiConfig loihi_config = {
        .num_cores = 128,
        .neurons_per_core = 1024,
        .synapses_per_core = 4096,
        .learning_rate = 0.01,
        .enable_adaptive_threshold = 1,
        .enable_structural_plasticity = 1
    };
    
    if (loihi_ode_init(&loihi, n, &loihi_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_loihi = (double*)malloc(n * sizeof(double));
        if (y_loihi) {
            if (loihi_ode_solve(&loihi, f, t0, t_end, y0_copy, h, params, y_loihi) == 0) {
                end = clock();
                
                results->loihi_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->loihi_steps = (size_t)((t_end - t0) / h);
                results->loihi_spikes = loihi.spikes_generated;
                results->loihi_error = compute_error(y_loihi, exact_solution, n);
                results->loihi_accuracy = compute_accuracy(y_loihi, exact_solution, n);
            }
            free(y_loihi);
        }
        
        loihi_ode_free(&loihi);
        free(y0_copy);
    }
    
    // Test BrainChips
    BrainChipsSolver brainchips;
    BrainChipsConfig bc_config = {
        .num_neurons = 100000,
        .num_synapses = 1000000,
        .neuron_leak_rate = 0.1,
        .enable_event_driven = 1,
        .enable_sparse_representation = 1
    };
    
    if (brainchips_ode_init(&brainchips, n, &bc_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_bc = (double*)malloc(n * sizeof(double));
        if (y_bc) {
            if (brainchips_ode_solve(&brainchips, f, t0, t_end, y0_copy, h, params, y_bc) == 0) {
                end = clock();
                
                results->brainchips_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->brainchips_steps = (size_t)((t_end - t0) / h);
                results->brainchips_events = brainchips.events_processed;
                results->brainchips_error = compute_error(y_bc, exact_solution, n);
                results->brainchips_accuracy = compute_accuracy(y_bc, exact_solution, n);
            }
            free(y_bc);
        }
        
        brainchips_ode_free(&brainchips);
        free(y0_copy);
    }
    
    // Test Racetrack (Parkin)
    RacetrackSolver racetrack;
    RacetrackConfig rt_config = {
        .num_tracks = 256,
        .domains_per_track = 64,
        .domain_wall_velocity = 100.0, // m/s
        .read_write_latency = 10.0, // 10 ns
        .enable_3d_stacking = 1
    };
    
    if (racetrack_ode_init(&racetrack, n, &rt_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_rt = (double*)malloc(n * sizeof(double));
        if (y_rt) {
            if (racetrack_ode_solve(&racetrack, f, t0, t_end, y0_copy, h, params, y_rt) == 0) {
                end = clock();
                
                results->racetrack_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->racetrack_steps = (size_t)((t_end - t0) / h);
                results->racetrack_domain_movements = racetrack.domain_wall_movements;
                results->racetrack_error = compute_error(y_rt, exact_solution, n);
                results->racetrack_accuracy = compute_accuracy(y_rt, exact_solution, n);
            }
            free(y_rt);
        }
        
        racetrack_ode_free(&racetrack);
        free(y0_copy);
    }
    
    // Test Phase Change Memory (IBM)
    PCMSolver pcm;
    PCMConfig pcm_config = {
        .num_cells = 1024,
        .set_resistance = 1e3, // 1 kOhm
        .reset_resistance = 1e6, // 1 MOhm
        .programming_time = 100.0, // 100 ns
        .enable_multi_level = 1
    };
    
    if (pcm_ode_init(&pcm, n, &pcm_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_pcm = (double*)malloc(n * sizeof(double));
        if (y_pcm) {
            if (pcm_ode_solve(&pcm, f, t0, t_end, y0_copy, h, params, y_pcm) == 0) {
                end = clock();
                
                results->pcm_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->pcm_steps = (size_t)((t_end - t0) / h);
                results->pcm_phase_transitions = pcm.phase_transitions;
                results->pcm_error = compute_error(y_pcm, exact_solution, n);
                results->pcm_accuracy = compute_accuracy(y_pcm, exact_solution, n);
            }
            free(y_pcm);
        }
        
        pcm_ode_free(&pcm);
        free(y0_copy);
    }
    
    // Test Lyric (MIT)
    LyricSolver lyric;
    LyricConfig lyric_config = {
        .num_probabilistic_units = 256,
        .random_bit_generators = 64,
        .probability_precision = 32, // 32 bits
        .enable_bayesian_inference = 1,
        .enable_markov_chain = 1
    };
    
    if (lyric_ode_init(&lyric, n, &lyric_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_lyric = (double*)malloc(n * sizeof(double));
        if (y_lyric) {
            if (lyric_ode_solve(&lyric, f, t0, t_end, y0_copy, h, params, y_lyric) == 0) {
                end = clock();
                
                results->lyric_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->lyric_steps = (size_t)((t_end - t0) / h);
                results->lyric_samples = lyric.samples_generated;
                results->lyric_error = compute_error(y_lyric, exact_solution, n);
                results->lyric_accuracy = compute_accuracy(y_lyric, exact_solution, n);
            }
            free(y_lyric);
        }
        
        lyric_ode_free(&lyric);
        free(y0_copy);
    }
    
    // Test HW Bayesian Networks (Chandra)
    HWBayesianSolver hw_bayesian;
    HWBayesianConfig hwb_config = {
        .num_nodes = 256,
        .num_edges = 512,
        .inference_engine_size = 1024,
        .enable_parallel_inference = 1,
        .enable_approximate_inference = 0
    };
    
    if (hw_bayesian_ode_init(&hw_bayesian, n, &hwb_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_hwb = (double*)malloc(n * sizeof(double));
        if (y_hwb) {
            if (hw_bayesian_ode_solve(&hw_bayesian, f, t0, t_end, y0_copy, h, params, y_hwb) == 0) {
                end = clock();
                
                results->hw_bayesian_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->hw_bayesian_steps = (size_t)((t_end - t0) / h);
                results->hw_bayesian_inference_ops = hw_bayesian.inference_operations;
                results->hw_bayesian_error = compute_error(y_hwb, exact_solution, n);
                results->hw_bayesian_accuracy = compute_accuracy(y_hwb, exact_solution, n);
            }
            free(y_hwb);
        }
        
        hw_bayesian_ode_free(&hw_bayesian);
        free(y0_copy);
    }
    
    // Test Semantic Lexographic Binary Search (Chandra & Chandra)
    SemanticLexoBSSolver semantic_lexo_bs;
    SemanticLexoBSConfig slbs_config = {
        .num_threads = 512,
        .semantic_tree_depth = 10,
        .lexographic_order_size = 1024,
        .enable_tail_recursion = 1,
        .enable_semantic_caching = 1
    };
    
    if (semantic_lexo_bs_ode_init(&semantic_lexo_bs, n, &slbs_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_slbs = (double*)malloc(n * sizeof(double));
        if (y_slbs) {
            if (semantic_lexo_bs_ode_solve(&semantic_lexo_bs, f, t0, t_end, y0_copy, h, params, y_slbs) == 0) {
                end = clock();
                
                results->semantic_lexo_bs_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->semantic_lexo_bs_steps = (size_t)((t_end - t0) / h);
                results->semantic_lexo_bs_nodes_searched = semantic_lexo_bs.nodes_searched;
                results->semantic_lexo_bs_error = compute_error(y_slbs, exact_solution, n);
                results->semantic_lexo_bs_accuracy = compute_accuracy(y_slbs, exact_solution, n);
            }
            free(y_slbs);
        }
        
        semantic_lexo_bs_ode_free(&semantic_lexo_bs);
        free(y0_copy);
    }
    
    // Test Kernelized SPS Binary Search (Chandra, Shyamal)
    KernelizedSPSBSSolver kernelized_sps_bs;
    KernelizedSPSBSConfig kspsbs_config = {
        .num_kernels = 64,
        .semantic_dim = 128,
        .pragmatic_dim = 128,
        .syntactic_dim = 128,
        .kernel_bandwidth = 1.0,
        .enable_kernel_caching = 1
    };
    
    if (kernelized_sps_bs_ode_init(&kernelized_sps_bs, n, &kspsbs_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_kspsbs = (double*)malloc(n * sizeof(double));
        if (y_kspsbs) {
            if (kernelized_sps_bs_ode_solve(&kernelized_sps_bs, f, t0, t_end, y0_copy, h, params, y_kspsbs) == 0) {
                end = clock();
                
                results->kernelized_sps_bs_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->kernelized_sps_bs_steps = (size_t)((t_end - t0) / h);
                results->kernelized_sps_bs_kernel_evals = kernelized_sps_bs.kernel_evaluations;
                results->kernelized_sps_bs_error = compute_error(y_kspsbs, exact_solution, n);
                results->kernelized_sps_bs_accuracy = compute_accuracy(y_kspsbs, exact_solution, n);
            }
            free(y_kspsbs);
        }
        
        kernelized_sps_bs_ode_free(&kernelized_sps_bs);
        free(y0_copy);
    }
    
    // Test Spiralizer with Chord Algorithm (Chandra, Shyamal)
    SpiralizerChordSolver spiralizer_chord;
    SpiralizerChordConfig sc_config = {
        .num_nodes = 256,
        .finger_table_size = 8, // log2(256)
        .hash_table_size = 1024,
        .hash_collision_rate = 0.1,
        .enable_morris_hashing = 1,
        .enable_spiral_traversal = 1,
        .spiral_radius = 2.0
    };
    
    if (spiralizer_chord_ode_init(&spiralizer_chord, n, &sc_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_sc = (double*)malloc(n * sizeof(double));
        if (y_sc) {
            if (spiralizer_chord_ode_solve(&spiralizer_chord, f, t0, t_end, y0_copy, h, params, y_sc) == 0) {
                end = clock();
                
                results->spiralizer_chord_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->spiralizer_chord_steps = (size_t)((t_end - t0) / h);
                results->spiralizer_chord_collisions = spiralizer_chord.hash_collisions;
                results->spiralizer_chord_spiral_steps = spiralizer_chord.spiral_steps;
                results->spiralizer_chord_error = compute_error(y_sc, exact_solution, n);
                results->spiralizer_chord_accuracy = compute_accuracy(y_sc, exact_solution, n);
            }
            free(y_sc);
        }
        
        spiralizer_chord_ode_free(&spiralizer_chord);
        free(y0_copy);
    }
    
    // Test Lattice Architecture (Waterfront variation - Chandra, Shyamal)
    LatticeWaterfrontSolver lattice_waterfront;
    LatticeWaterfrontConfig lw_config = {
        .lattice_dimensions = 4,
        .nodes_per_dimension = 16,
        .waterfront_size = 256,
        .lattice_spacing = 1.0,
        .enable_waterfront_buffering = 1,
        .enable_lattice_routing = 1,
        .routing_latency = 5.0 // 5 ns
    };
    
    if (lattice_waterfront_ode_init(&lattice_waterfront, n, &lw_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_lw = (double*)malloc(n * sizeof(double));
        if (y_lw) {
            if (lattice_waterfront_ode_solve(&lattice_waterfront, f, t0, t_end, y0_copy, h, params, y_lw) == 0) {
                end = clock();
                
                results->lattice_waterfront_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->lattice_waterfront_steps = (size_t)((t_end - t0) / h);
                results->lattice_waterfront_routing_ops = lattice_waterfront.routing_operations;
                results->lattice_waterfront_waterfront_ops = lattice_waterfront.waterfront_operations;
                results->lattice_waterfront_error = compute_error(y_lw, exact_solution, n);
                results->lattice_waterfront_accuracy = compute_accuracy(y_lw, exact_solution, n);
            }
            free(y_lw);
        }
        
        lattice_waterfront_ode_free(&lattice_waterfront);
        free(y0_copy);
    }
    
    // Test Multiple-Search Representation Tree Algorithm
    MultipleSearchTreeSolver multiple_search_tree;
    MultipleSearchTreeConfig mst_config = {
        .max_tree_depth = 100,
        .max_nodes = 10000,
        .num_search_strategies = 4,
        .enable_bfs = 1,
        .enable_dfs = 1,
        .enable_astar = 1,
        .enable_best_first = 1,
        .heuristic_weight = 1.0,
        .representation_switch_threshold = 0.1,
        .enable_tree_representation = 1,
        .enable_graph_representation = 1
    };
    
    if (multiple_search_tree_ode_init(&multiple_search_tree, n, &mst_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_mst = (double*)malloc(n * sizeof(double));
        if (y_mst) {
            if (multiple_search_tree_ode_solve(&multiple_search_tree, f, t0, t_end, y0_copy, h, params, y_mst) == 0) {
                end = clock();
                
                results->multiple_search_tree_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->multiple_search_tree_steps = (size_t)((t_end - t0) / h);
                results->multiple_search_tree_nodes_expanded = multiple_search_tree.nodes_expanded;
                results->multiple_search_tree_nodes_generated = multiple_search_tree.nodes_generated;
                results->multiple_search_tree_error = compute_error(y_mst, exact_solution, n);
                results->multiple_search_tree_accuracy = compute_accuracy(y_mst, exact_solution, n);
            }
            free(y_mst);
        }
        
        multiple_search_tree_ode_free(&multiple_search_tree);
        free(y0_copy);
    }
    
    // Test Directed Diffusion with Manhattan Distance (Chandra, Shyamal)
    // Inspired by Deborah Estrin and Ramesh Govindan et al.
    // Temporarily disabled for stability - will be re-enabled after optimization
    /*
    DirectedDiffusionSolver directed_diffusion;
    DirectedDiffusionConfig dd_config = {
        .grid_size = 8,  // Small grid for stability
        .num_sources = 2,
        .num_sinks = 2,
        .diffusion_rate = 0.1,
        .manhattan_weight = 0.5,
        .flood_fill_threshold = 3.0,
        .enable_static_focus = 1,
        .enable_gradient_repair = 0,  // Disabled for stability
        .max_flood_iterations = 100,
        .interest_decay_rate = 0.01,
        .data_aggregation_rate = 0.1
    };
    
    if (directed_diffusion_ode_init(&directed_diffusion, n, &dd_config) == 0) {
        y0_copy = (double*)malloc(n * sizeof(double));
        memcpy(y0_copy, y0, n * sizeof(double));
        
        start = clock();
        double* y_dd = (double*)malloc(n * sizeof(double));
        if (y_dd) {
            if (directed_diffusion_ode_solve(&directed_diffusion, f, t0, t_end, y0_copy, h, params, y_dd) == 0) {
                end = clock();
                
                results->directed_diffusion_time = ((double)(end - start)) / CLOCKS_PER_SEC;
                results->directed_diffusion_steps = (size_t)((t_end - t0) / h);
                results->directed_diffusion_flood_iterations = directed_diffusion.flood_iterations;
                results->directed_diffusion_gradient_updates = directed_diffusion.gradient_updates;
                results->directed_diffusion_error = compute_error(y_dd, exact_solution, n);
                results->directed_diffusion_accuracy = compute_accuracy(y_dd, exact_solution, n);
            }
            free(y_dd);
        }
        
        directed_diffusion_ode_free(&directed_diffusion);
        free(y0_copy);
    }
    */
    
    free(t_out);
    free(y_out);
    
    return 0;
}

void print_comparison_results(const ComparisonResults* results) {
    if (!results) return;
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  METHOD COMPARISON: Standard + Parallel + Stacked Methods      ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ Method      │ Time (s) │ Steps │ Error      │ Accuracy          ║\n");
    printf("╠═════════════╪══════════╪═══════╪════════════╪═══════════════════╣\n");
    printf("║ Euler       │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->euler_time, results->euler_steps, results->euler_error, results->euler_accuracy * 100);
    printf("║ DDEuler     │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->ddeuler_time, results->ddeuler_steps, results->ddeuler_error, results->ddeuler_accuracy * 100);
    printf("║ RK3         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->rk3_time, results->rk3_steps, results->rk3_error, results->rk3_accuracy * 100);
    if (results->rk4_time > 0) {
        printf("║ RK4         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->rk4_time, results->rk4_steps, results->rk4_error, results->rk4_accuracy * 100);
    }
    printf("║ DDRK3       │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->ddrk3_time, results->ddrk3_steps, results->ddrk3_error, results->ddrk3_accuracy * 100);
    if (results->am1_time > 0) {
        printf("║ AM1         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->am1_time, results->am1_steps, results->am1_error, results->am1_accuracy * 100);
    }
    if (results->am2_time > 0) {
        printf("║ AM2         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->am2_time, results->am2_steps, results->am2_error, results->am2_accuracy * 100);
    }
    printf("║ AM3         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->am_time, results->am_steps, results->am_error, results->am_accuracy * 100);
    if (results->am4_time > 0) {
        printf("║ AM4         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->am4_time, results->am4_steps, results->am4_error, results->am4_accuracy * 100);
    }
    if (results->am5_time > 0) {
        printf("║ AM5         │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->am5_time, results->am5_steps, results->am5_error, results->am5_accuracy * 100);
    }
    printf("║ DDAM        │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->ddam_time, results->ddam_steps, results->ddam_error, results->ddam_accuracy * 100);
    if (results->parallel_rk3_time > 0) {
        printf("║ Parallel RK3 │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->parallel_rk3_time, results->parallel_rk3_steps, results->parallel_rk3_error, results->parallel_rk3_accuracy * 100);
        printf("║   (Speedup: %.2fx) │\n", results->speedup_rk3);
    }
    if (results->stacked_rk3_time > 0) {
        printf("║ Stacked RK3  │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->stacked_rk3_time, results->stacked_rk3_steps, results->stacked_rk3_error, results->stacked_rk3_accuracy * 100);
    }
    if (results->parallel_am_time > 0) {
        printf("║ Parallel AM  │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->parallel_am_time, results->parallel_am_steps, results->parallel_am_error, results->parallel_am_accuracy * 100);
        printf("║   (Speedup: %.2fx) │\n", results->speedup_am);
    }
    if (results->parallel_euler_time > 0) {
        printf("║ Parallel Euler│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->parallel_euler_time, results->parallel_euler_steps, results->parallel_euler_error, results->parallel_euler_accuracy * 100);
        printf("║   (Speedup: %.2fx) │\n", results->speedup_euler);
    }
    if (results->realtime_rk3_time > 0) {
        printf("║ Real-Time RK3  │ %8.6f │   N/A │ %10.6e │ %17.6f%% ║\n",
               results->realtime_rk3_time, results->realtime_rk3_error, results->realtime_rk3_accuracy * 100);
    }
    if (results->online_rk3_time > 0) {
        printf("║ Online RK3     │ %8.6f │   N/A │ %10.6e │ %17.6f%% ║\n",
               results->online_rk3_time, results->online_rk3_error, results->online_rk3_accuracy * 100);
    }
    if (results->dynamic_rk3_time > 0) {
        printf("║ Dynamic RK3    │ %8.6f │   N/A │ %10.6e │ %17.6f%% ║\n",
               results->dynamic_rk3_time, results->dynamic_rk3_error, results->dynamic_rk3_accuracy * 100);
    }
    if (results->nonlinear_ode_time > 0) {
        printf("║ Nonlinear ODE │ %8.6f │   N/A │ %10.6e │ %17.6f%% ║\n",
               results->nonlinear_ode_time, results->nonlinear_ode_error, results->nonlinear_ode_accuracy * 100);
    }
    if (results->karmarkar_time > 0) {
        printf("║ Karmarkar     │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->karmarkar_time, results->karmarkar_steps, results->karmarkar_error, results->karmarkar_accuracy * 100);
        printf("║   (Iterations: %zu) │\n", results->karmarkar_iterations);
    }
    if (results->distributed_datadriven_time > 0) {
        printf("║ Dist+DD Solver │ %8.6f │   N/A │       N/A │              N/A ║\n",
               results->distributed_datadriven_time);
    }
    if (results->online_datadriven_time > 0) {
        printf("║ Online+DD      │ %8.6f │   N/A │       N/A │              N/A ║\n",
               results->online_datadriven_time);
    }
    if (results->mapreduce_time > 0) {
        printf("║ Map/Reduce      │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->mapreduce_time, results->mapreduce_steps, results->mapreduce_error,
               results->mapreduce_accuracy * 100);
        printf("║   (Map: %.4fs, Reduce: %.4fs, Shuffle: %.4fs) │\n",
               results->mapreduce_map_time, results->mapreduce_reduce_time, results->mapreduce_shuffle_time);
        printf("║   (Cost: $%.4f) │\n", results->mapreduce_cost);
    }
    if (results->spark_time > 0) {
        printf("║ Spark           │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->spark_time, results->spark_steps, results->spark_error,
               results->spark_accuracy * 100);
        printf("║   (Map: %.4fs, Reduce: %.4fs, Shuffle: %.4fs) │\n",
               results->spark_map_time, results->spark_reduce_time, results->spark_shuffle_time);
        printf("║   (Cache Hit: %.2f%%, Cost: $%.4f) │\n",
               results->spark_cache_hit_rate * 100, results->spark_cost);
    }
    if (results->microgasjet_time > 0) {
        printf("║ Micro-Gas Jet   │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->microgasjet_time, results->microgasjet_steps, results->microgasjet_error,
               results->microgasjet_accuracy * 100);
        printf("║   (Flow Energy: %.4e J) │\n", results->microgasjet_flow_energy);
    }
    if (results->dataflow_time > 0) {
        printf("║ Dataflow (Arvind)│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->dataflow_time, results->dataflow_steps, results->dataflow_error,
               results->dataflow_accuracy * 100);
        printf("║   (Tokens: %zu, Matching: %.4fs) │\n",
               results->dataflow_tokens, results->dataflow_token_matching_time);
    }
    if (results->ace_time > 0) {
        printf("║ ACE (Turing)    │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->ace_time, results->ace_steps, results->ace_error,
               results->ace_accuracy * 100);
        printf("║   (Instructions: %zu, Memory: %.4fs) │\n",
               results->ace_instructions, results->ace_memory_time);
    }
    if (results->systolic_time > 0) {
        printf("║ Systolic Array  │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->systolic_time, results->systolic_steps, results->systolic_error,
               results->systolic_accuracy * 100);
        printf("║   (Comm Time: %.4fs) │\n", results->systolic_communication_time);
    }
    if (results->tpu_time > 0) {
        printf("║ TPU (Patterson) │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->tpu_time, results->tpu_steps, results->tpu_error,
               results->tpu_accuracy * 100);
        printf("║   (Matrix Ops: %zu, BW Util: %.2f%%) │\n",
               results->tpu_matrix_ops, results->tpu_bandwidth_utilization * 100);
    }
    if (results->gpu_cuda_time > 0) {
        printf("║ GPU (CUDA)      │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->gpu_cuda_time, results->gpu_cuda_steps, results->gpu_cuda_error,
               results->gpu_cuda_accuracy * 100);
        printf("║   (Kernels: %zu, Mem Xfer: %.4fs) │\n",
               results->gpu_cuda_kernel_launches, results->gpu_cuda_memory_transfer_time);
    }
    if (results->gpu_metal_time > 0) {
        printf("║ GPU (Metal)     │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->gpu_metal_time, results->gpu_metal_steps, results->gpu_metal_error,
               results->gpu_metal_accuracy * 100);
    }
    if (results->gpu_vulkan_time > 0) {
        printf("║ GPU (Vulkan)    │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->gpu_vulkan_time, results->gpu_vulkan_steps, results->gpu_vulkan_error,
               results->gpu_vulkan_accuracy * 100);
    }
    if (results->gpu_amd_time > 0) {
        printf("║ GPU (AMD)       │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->gpu_amd_time, results->gpu_amd_steps, results->gpu_amd_error,
               results->gpu_amd_accuracy * 100);
    }
    if (results->massively_threaded_time > 0) {
        printf("║ Massively-Threaded│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->massively_threaded_time, results->massively_threaded_steps, results->massively_threaded_error,
               results->massively_threaded_accuracy * 100);
        printf("║   (Nodes Expanded: %zu) │\n", results->massively_threaded_nodes_expanded);
    }
    if (results->starr_time > 0) {
        printf("║ STARR (Chandra) │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->starr_time, results->starr_steps, results->starr_error,
               results->starr_accuracy * 100);
        printf("║   (Semantic: %zu, Associative: %zu) │\n",
               results->starr_semantic_hits, results->starr_associative_hits);
    }
    if (results->truenorth_time > 0) {
        printf("║ TrueNorth (IBM)  │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->truenorth_time, results->truenorth_steps, results->truenorth_error,
               results->truenorth_accuracy * 100);
        printf("║   (Spikes: %zu, Energy: %.4e J) │\n",
               results->truenorth_spikes, results->truenorth_energy);
    }
    if (results->loihi_time > 0) {
        printf("║ Loihi (Intel)   │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->loihi_time, results->loihi_steps, results->loihi_error,
               results->loihi_accuracy * 100);
        printf("║   (Spikes: %zu) │\n", results->loihi_spikes);
    }
    if (results->brainchips_time > 0) {
        printf("║ BrainChips      │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->brainchips_time, results->brainchips_steps, results->brainchips_error,
               results->brainchips_accuracy * 100);
        printf("║   (Events: %zu) │\n", results->brainchips_events);
    }
    if (results->racetrack_time > 0) {
        printf("║ Racetrack (Parkin)│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->racetrack_time, results->racetrack_steps, results->racetrack_error,
               results->racetrack_accuracy * 100);
        printf("║   (Domain Movements: %zu) │\n", results->racetrack_domain_movements);
    }
    if (results->pcm_time > 0) {
        printf("║ Phase Change Mem│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->pcm_time, results->pcm_steps, results->pcm_error,
               results->pcm_accuracy * 100);
        printf("║   (Phase Transitions: %zu) │\n", results->pcm_phase_transitions);
    }
    if (results->lyric_time > 0) {
        printf("║ Lyric (MIT)     │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->lyric_time, results->lyric_steps, results->lyric_error,
               results->lyric_accuracy * 100);
        printf("║   (Samples: %zu) │\n", results->lyric_samples);
    }
    if (results->hw_bayesian_time > 0) {
        printf("║ HW Bayesian     │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->hw_bayesian_time, results->hw_bayesian_steps, results->hw_bayesian_error,
               results->hw_bayesian_accuracy * 100);
        printf("║   (Inference Ops: %zu) │\n", results->hw_bayesian_inference_ops);
    }
    if (results->semantic_lexo_bs_time > 0) {
        printf("║ Semantic Lexo BS│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->semantic_lexo_bs_time, results->semantic_lexo_bs_steps, results->semantic_lexo_bs_error,
               results->semantic_lexo_bs_accuracy * 100);
        printf("║   (Nodes Searched: %zu) │\n", results->semantic_lexo_bs_nodes_searched);
    }
    if (results->kernelized_sps_bs_time > 0) {
        printf("║ Kernelized SPS BS│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->kernelized_sps_bs_time, results->kernelized_sps_bs_steps, results->kernelized_sps_bs_error,
               results->kernelized_sps_bs_accuracy * 100);
        printf("║   (Kernel Evals: %zu) │\n", results->kernelized_sps_bs_kernel_evals);
    }
    if (results->spiralizer_chord_time > 0) {
        printf("║ Spiralizer Chord│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->spiralizer_chord_time, results->spiralizer_chord_steps, results->spiralizer_chord_error,
               results->spiralizer_chord_accuracy * 100);
        printf("║   (Collisions: %zu, Spiral Steps: %zu) │\n",
               results->spiralizer_chord_collisions, results->spiralizer_chord_spiral_steps);
    }
    if (results->lattice_waterfront_time > 0) {
        printf("║ Lattice Waterfront│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->lattice_waterfront_time, results->lattice_waterfront_steps, results->lattice_waterfront_error,
               results->lattice_waterfront_accuracy * 100);
        printf("║   (Routing Ops: %zu, Waterfront Ops: %zu) │\n",
               results->lattice_waterfront_routing_ops, results->lattice_waterfront_waterfront_ops);
    }
    if (results->multiple_search_tree_time > 0) {
        printf("║ Multiple-Search Tree│ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->multiple_search_tree_time, results->multiple_search_tree_steps, results->multiple_search_tree_error,
               results->multiple_search_tree_accuracy * 100);
        printf("║   (Expanded: %zu, Generated: %zu) │\n",
               results->multiple_search_tree_nodes_expanded, results->multiple_search_tree_nodes_generated);
    }
    if (results->directed_diffusion_time > 0) {
        printf("║ Directed Diffusion │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
               results->directed_diffusion_time, results->directed_diffusion_steps, results->directed_diffusion_error,
               results->directed_diffusion_accuracy * 100);
        printf("║   (Flood: %zu, Gradient: %zu) │\n",
               results->directed_diffusion_flood_iterations, results->directed_diffusion_gradient_updates);
    }
    printf("╚═════════════╧══════════╧═══════╧════════════╧═══════════════════╝\n");
    printf("\n");
    
    // Find best method
    const char* best_time = "Euler";
    double min_time = results->euler_time;
    if (results->ddeuler_time > 0 && results->ddeuler_time < min_time) {
        min_time = results->ddeuler_time;
        best_time = "DDEuler";
    }
    if (results->rk3_time > 0 && results->rk3_time < min_time) {
        min_time = results->rk3_time;
        best_time = "RK3";
    }
    if (results->rk4_time > 0 && results->rk4_time < min_time) {
        min_time = results->rk4_time;
        best_time = "RK4";
    }
    if (results->ddrk3_time > 0 && results->ddrk3_time < min_time) {
        min_time = results->ddrk3_time;
        best_time = "DDRK3";
    }
    if (results->am1_time > 0 && results->am1_time < min_time) {
        min_time = results->am1_time;
        best_time = "AM1";
    }
    if (results->am2_time > 0 && results->am2_time < min_time) {
        min_time = results->am2_time;
        best_time = "AM2";
    }
    if (results->am_time > 0 && results->am_time < min_time) {
        min_time = results->am_time;
        best_time = "AM3";
    }
    if (results->am4_time > 0 && results->am4_time < min_time) {
        min_time = results->am4_time;
        best_time = "AM4";
    }
    if (results->am5_time > 0 && results->am5_time < min_time) {
        min_time = results->am5_time;
        best_time = "AM5";
    }
    if (results->ddam_time > 0 && results->ddam_time < min_time) {
        min_time = results->ddam_time;
        best_time = "DDAM";
    }
    
    const char* best_accuracy = "Euler";
    double max_accuracy = results->euler_accuracy;
    if (results->ddeuler_accuracy > max_accuracy) {
        max_accuracy = results->ddeuler_accuracy;
        best_accuracy = "DDEuler";
    }
    if (results->rk3_accuracy > max_accuracy) {
        max_accuracy = results->rk3_accuracy;
        best_accuracy = "RK3";
    }
    if (results->rk4_accuracy > max_accuracy) {
        max_accuracy = results->rk4_accuracy;
        best_accuracy = "RK4";
    }
    if (results->ddrk3_accuracy > max_accuracy) {
        max_accuracy = results->ddrk3_accuracy;
        best_accuracy = "DDRK3";
    }
    if (results->am1_accuracy > max_accuracy) {
        max_accuracy = results->am1_accuracy;
        best_accuracy = "AM1";
    }
    if (results->am2_accuracy > max_accuracy) {
        max_accuracy = results->am2_accuracy;
        best_accuracy = "AM2";
    }
    if (results->am_accuracy > max_accuracy) {
        max_accuracy = results->am_accuracy;
        best_accuracy = "AM3";
    }
    if (results->am4_accuracy > max_accuracy) {
        max_accuracy = results->am4_accuracy;
        best_accuracy = "AM4";
    }
    if (results->am5_accuracy > max_accuracy) {
        max_accuracy = results->am5_accuracy;
        best_accuracy = "AM5";
    }
    if (results->ddam_accuracy > max_accuracy) {
        max_accuracy = results->ddam_accuracy;
        best_accuracy = "DDAM";
    }
    if (results->directed_diffusion_time > 0 && results->directed_diffusion_time < min_time) {
        min_time = results->directed_diffusion_time;
        best_time = "Directed Diffusion";
    }
    if (results->directed_diffusion_accuracy > max_accuracy) {
        max_accuracy = results->directed_diffusion_accuracy;
        best_accuracy = "Directed Diffusion";
    }
    if (results->karmarkar_time > 0 && results->karmarkar_time < min_time) {
        min_time = results->karmarkar_time;
        best_time = "Karmarkar";
    }
    if (results->karmarkar_accuracy > max_accuracy) {
        max_accuracy = results->karmarkar_accuracy;
        best_accuracy = "Karmarkar";
    }
    if (results->mapreduce_time > 0 && results->mapreduce_time < min_time) {
        min_time = results->mapreduce_time;
        best_time = "MapReduce";
    }
    if (results->mapreduce_accuracy > max_accuracy) {
        max_accuracy = results->mapreduce_accuracy;
        best_accuracy = "MapReduce";
    }
    if (results->spark_time > 0 && results->spark_time < min_time) {
        min_time = results->spark_time;
        best_time = "Spark";
    }
    if (results->spark_accuracy > max_accuracy) {
        max_accuracy = results->spark_accuracy;
        best_accuracy = "Spark";
    }
    if (results->microgasjet_time > 0 && results->microgasjet_time < min_time) {
        min_time = results->microgasjet_time;
        best_time = "Micro-Gas Jet";
    }
    if (results->microgasjet_accuracy > max_accuracy) {
        max_accuracy = results->microgasjet_accuracy;
        best_accuracy = "Micro-Gas Jet";
    }
    if (results->dataflow_time > 0 && results->dataflow_time < min_time) {
        min_time = results->dataflow_time;
        best_time = "Dataflow";
    }
    if (results->dataflow_accuracy > max_accuracy) {
        max_accuracy = results->dataflow_accuracy;
        best_accuracy = "Dataflow";
    }
    if (results->gpu_cuda_time > 0 && results->gpu_cuda_time < min_time) {
        min_time = results->gpu_cuda_time;
        best_time = "GPU (CUDA)";
    }
    if (results->gpu_cuda_accuracy > max_accuracy) {
        max_accuracy = results->gpu_cuda_accuracy;
        best_accuracy = "GPU (CUDA)";
    }
    if (results->tpu_time > 0 && results->tpu_time < min_time) {
        min_time = results->tpu_time;
        best_time = "TPU";
    }
    if (results->tpu_accuracy > max_accuracy) {
        max_accuracy = results->tpu_accuracy;
        best_accuracy = "TPU";
    }
    if (results->massively_threaded_time > 0 && results->massively_threaded_time < min_time) {
        min_time = results->massively_threaded_time;
        best_time = "Massively-Threaded (Korf)";
    }
    if (results->massively_threaded_accuracy > max_accuracy) {
        max_accuracy = results->massively_threaded_accuracy;
        best_accuracy = "Massively-Threaded (Korf)";
    }
    if (results->starr_time > 0 && results->starr_time < min_time) {
        min_time = results->starr_time;
        best_time = "STARR (Chandra)";
    }
    if (results->starr_accuracy > max_accuracy) {
        max_accuracy = results->starr_accuracy;
        best_accuracy = "STARR (Chandra)";
    }
    if (results->truenorth_time > 0 && results->truenorth_time < min_time) {
        min_time = results->truenorth_time;
        best_time = "TrueNorth (IBM)";
    }
    if (results->truenorth_accuracy > max_accuracy) {
        max_accuracy = results->truenorth_accuracy;
        best_accuracy = "TrueNorth (IBM)";
    }
    if (results->loihi_time > 0 && results->loihi_time < min_time) {
        min_time = results->loihi_time;
        best_time = "Loihi (Intel)";
    }
    if (results->loihi_accuracy > max_accuracy) {
        max_accuracy = results->loihi_accuracy;
        best_accuracy = "Loihi (Intel)";
    }
    if (results->brainchips_time > 0 && results->brainchips_time < min_time) {
        min_time = results->brainchips_time;
        best_time = "BrainChips";
    }
    if (results->brainchips_accuracy > max_accuracy) {
        max_accuracy = results->brainchips_accuracy;
        best_accuracy = "BrainChips";
    }
    if (results->racetrack_time > 0 && results->racetrack_time < min_time) {
        min_time = results->racetrack_time;
        best_time = "Racetrack (Parkin)";
    }
    if (results->racetrack_accuracy > max_accuracy) {
        max_accuracy = results->racetrack_accuracy;
        best_accuracy = "Racetrack (Parkin)";
    }
    if (results->pcm_time > 0 && results->pcm_time < min_time) {
        min_time = results->pcm_time;
        best_time = "Phase Change Memory";
    }
    if (results->pcm_accuracy > max_accuracy) {
        max_accuracy = results->pcm_accuracy;
        best_accuracy = "Phase Change Memory";
    }
    if (results->lyric_time > 0 && results->lyric_time < min_time) {
        min_time = results->lyric_time;
        best_time = "Lyric (MIT)";
    }
    if (results->lyric_accuracy > max_accuracy) {
        max_accuracy = results->lyric_accuracy;
        best_accuracy = "Lyric (MIT)";
    }
    if (results->hw_bayesian_time > 0 && results->hw_bayesian_time < min_time) {
        min_time = results->hw_bayesian_time;
        best_time = "HW Bayesian (Chandra)";
    }
    if (results->hw_bayesian_accuracy > max_accuracy) {
        max_accuracy = results->hw_bayesian_accuracy;
        best_accuracy = "HW Bayesian (Chandra)";
    }
    if (results->semantic_lexo_bs_time > 0 && results->semantic_lexo_bs_time < min_time) {
        min_time = results->semantic_lexo_bs_time;
        best_time = "Semantic Lexo BS (Chandra & Chandra)";
    }
    if (results->semantic_lexo_bs_accuracy > max_accuracy) {
        max_accuracy = results->semantic_lexo_bs_accuracy;
        best_accuracy = "Semantic Lexo BS (Chandra & Chandra)";
    }
    if (results->kernelized_sps_bs_time > 0 && results->kernelized_sps_bs_time < min_time) {
        min_time = results->kernelized_sps_bs_time;
        best_time = "Kernelized SPS BS (Chandra, Shyamal)";
    }
    if (results->kernelized_sps_bs_accuracy > max_accuracy) {
        max_accuracy = results->kernelized_sps_bs_accuracy;
        best_accuracy = "Kernelized SPS BS (Chandra, Shyamal)";
    }
    
    printf("🏆 Best Performance: %s (%.6f seconds)\n", best_time, min_time);
    printf("🎯 Best Accuracy: %s (%.6f%%)\n", best_accuracy, max_accuracy * 100);
    printf("\n");
}

int export_comparison_csv(const char* filename, const ComparisonResults* results) {
    if (!filename || !results) {
        return -1;
    }
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        return -1;
    }
    
    fprintf(fp, "Method,Time(s),Steps,Error,Accuracy(%%),Speedup,Workers\n");
    fprintf(fp, "Euler,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->euler_time, results->euler_steps, results->euler_error, results->euler_accuracy * 100);
    fprintf(fp, "DDEuler,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->ddeuler_time, results->ddeuler_steps, results->ddeuler_error, results->ddeuler_accuracy * 100);
    fprintf(fp, "RK3,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->rk3_time, results->rk3_steps, results->rk3_error, results->rk3_accuracy * 100);
    if (results->rk4_time > 0) {
        fprintf(fp, "RK4,%.6f,%zu,%.6e,%.6f,1.00,1\n",
                results->rk4_time, results->rk4_steps, results->rk4_error, results->rk4_accuracy * 100);
    }
    fprintf(fp, "DDRK3,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->ddrk3_time, results->ddrk3_steps, results->ddrk3_error, results->ddrk3_accuracy * 100);
    if (results->am1_time > 0) {
        fprintf(fp, "AM1,%.6f,%zu,%.6e,%.6f,1.00,1\n",
                results->am1_time, results->am1_steps, results->am1_error, results->am1_accuracy * 100);
    }
    if (results->am2_time > 0) {
        fprintf(fp, "AM2,%.6f,%zu,%.6e,%.6f,1.00,1\n",
                results->am2_time, results->am2_steps, results->am2_error, results->am2_accuracy * 100);
    }
    fprintf(fp, "AM3,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->am_time, results->am_steps, results->am_error, results->am_accuracy * 100);
    if (results->am4_time > 0) {
        fprintf(fp, "AM4,%.6f,%zu,%.6e,%.6f,1.00,1\n",
                results->am4_time, results->am4_steps, results->am4_error, results->am4_accuracy * 100);
    }
    if (results->am5_time > 0) {
        fprintf(fp, "AM5,%.6f,%zu,%.6e,%.6f,1.00,1\n",
                results->am5_time, results->am5_steps, results->am5_error, results->am5_accuracy * 100);
    }
    fprintf(fp, "DDAM,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->ddam_time, results->ddam_steps, results->ddam_error, results->ddam_accuracy * 100);
    if (results->parallel_rk3_time > 0) {
        fprintf(fp, "Parallel_RK3,%.6f,%zu,%.6e,%.6f,%.2f,%zu\n",
                results->parallel_rk3_time, results->parallel_rk3_steps, results->parallel_rk3_error,
                results->parallel_rk3_accuracy * 100, results->speedup_rk3, results->num_workers);
    }
    if (results->stacked_rk3_time > 0) {
        fprintf(fp, "Stacked_RK3,%.6f,%zu,%.6e,%.6f,1.00,%zu\n",
                results->stacked_rk3_time, results->stacked_rk3_steps, results->stacked_rk3_error,
                results->stacked_rk3_accuracy * 100, results->num_workers);
    }
    if (results->parallel_am_time > 0) {
        fprintf(fp, "Parallel_AM,%.6f,%zu,%.6e,%.6f,%.2f,%zu\n",
                results->parallel_am_time, results->parallel_am_steps, results->parallel_am_error,
                results->parallel_am_accuracy * 100, results->speedup_am, results->num_workers);
    }
    if (results->parallel_euler_time > 0) {
        fprintf(fp, "Parallel_Euler,%.6f,%zu,%.6e,%.6f,%.2f,%zu\n",
                results->parallel_euler_time, results->parallel_euler_steps, results->parallel_euler_error,
                results->parallel_euler_accuracy * 100, results->speedup_euler, results->num_workers);
    }
    if (results->realtime_rk3_time > 0) {
        fprintf(fp, "RealTime_RK3,%.6f,0,%.6e,%.6f,1.00,1\n",
                results->realtime_rk3_time, results->realtime_rk3_error,
                results->realtime_rk3_accuracy * 100);
    }
    if (results->online_rk3_time > 0) {
        fprintf(fp, "Online_RK3,%.6f,0,%.6e,%.6f,1.00,1\n",
                results->online_rk3_time, results->online_rk3_error,
                results->online_rk3_accuracy * 100);
    }
    if (results->dynamic_rk3_time > 0) {
        fprintf(fp, "Dynamic_RK3,%.6f,0,%.6e,%.6f,1.00,1\n",
                results->dynamic_rk3_time, results->dynamic_rk3_error,
                results->dynamic_rk3_accuracy * 100);
    }
    if (results->nonlinear_ode_time > 0) {
        fprintf(fp, "Nonlinear_ODE,%.6f,0,%.6e,%.6f,1.00,1\n",
               results->nonlinear_ode_time, results->nonlinear_ode_error, results->nonlinear_ode_accuracy * 100);
    }
    if (results->karmarkar_time > 0) {
        fprintf(fp, "Karmarkar,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->karmarkar_time, results->karmarkar_steps, results->karmarkar_error,
               results->karmarkar_accuracy * 100);
    }
    if (results->mapreduce_time > 0) {
        fprintf(fp, "MapReduce,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->mapreduce_time, results->mapreduce_steps, results->mapreduce_error,
               results->mapreduce_accuracy * 100);
        fprintf(fp, "MapReduce_Map,%.6f,0,0.0,0.0,1.00,1\n", results->mapreduce_map_time);
        fprintf(fp, "MapReduce_Reduce,%.6f,0,0.0,0.0,1.00,1\n", results->mapreduce_reduce_time);
        fprintf(fp, "MapReduce_Shuffle,%.6f,0,0.0,0.0,1.00,1\n", results->mapreduce_shuffle_time);
        fprintf(fp, "MapReduce_Cost,%.6f,0,0.0,0.0,1.00,1\n", results->mapreduce_cost);
    }
    if (results->spark_time > 0) {
        fprintf(fp, "Spark,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->spark_time, results->spark_steps, results->spark_error,
               results->spark_accuracy * 100);
        fprintf(fp, "Spark_Map,%.6f,0,0.0,0.0,1.00,1\n", results->spark_map_time);
        fprintf(fp, "Spark_Reduce,%.6f,0,0.0,0.0,1.00,1\n", results->spark_reduce_time);
        fprintf(fp, "Spark_Shuffle,%.6f,0,0.0,0.0,1.00,1\n", results->spark_shuffle_time);
        fprintf(fp, "Spark_CacheHitRate,%.6f,0,0.0,0.0,1.00,1\n", results->spark_cache_hit_rate * 100);
        fprintf(fp, "Spark_Cost,%.6f,0,0.0,0.0,1.00,1\n", results->spark_cost);
    }
    if (results->microgasjet_time > 0) {
        fprintf(fp, "MicroGasJet,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->microgasjet_time, results->microgasjet_steps, results->microgasjet_error,
               results->microgasjet_accuracy * 100);
    }
    if (results->dataflow_time > 0) {
        fprintf(fp, "Dataflow_Arvind,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->dataflow_time, results->dataflow_steps, results->dataflow_error,
               results->dataflow_accuracy * 100);
    }
    if (results->ace_time > 0) {
        fprintf(fp, "ACE_Turing,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->ace_time, results->ace_steps, results->ace_error,
               results->ace_accuracy * 100);
    }
    if (results->systolic_time > 0) {
        fprintf(fp, "Systolic_Array,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->systolic_time, results->systolic_steps, results->systolic_error,
               results->systolic_accuracy * 100);
    }
    if (results->tpu_time > 0) {
        fprintf(fp, "TPU_Patterson,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->tpu_time, results->tpu_steps, results->tpu_error,
               results->tpu_accuracy * 100);
    }
    if (results->gpu_cuda_time > 0) {
        fprintf(fp, "GPU_CUDA,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->gpu_cuda_time, results->gpu_cuda_steps, results->gpu_cuda_error,
               results->gpu_cuda_accuracy * 100);
    }
    if (results->gpu_metal_time > 0) {
        fprintf(fp, "GPU_Metal,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->gpu_metal_time, results->gpu_metal_steps, results->gpu_metal_error,
               results->gpu_metal_accuracy * 100);
    }
    if (results->gpu_vulkan_time > 0) {
        fprintf(fp, "GPU_Vulkan,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->gpu_vulkan_time, results->gpu_vulkan_steps, results->gpu_vulkan_error,
               results->gpu_vulkan_accuracy * 100);
    }
    if (results->gpu_amd_time > 0) {
        fprintf(fp, "GPU_AMD,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->gpu_amd_time, results->gpu_amd_steps, results->gpu_amd_error,
               results->gpu_amd_accuracy * 100);
    }
    if (results->massively_threaded_time > 0) {
        fprintf(fp, "Massively_Threaded,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->massively_threaded_time, results->massively_threaded_steps, results->massively_threaded_error,
               results->massively_threaded_accuracy * 100);
    }
    if (results->starr_time > 0) {
        fprintf(fp, "STARR_Chandra,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->starr_time, results->starr_steps, results->starr_error,
               results->starr_accuracy * 100);
    }
    if (results->truenorth_time > 0) {
        fprintf(fp, "TrueNorth_IBM,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->truenorth_time, results->truenorth_steps, results->truenorth_error,
               results->truenorth_accuracy * 100);
    }
    if (results->loihi_time > 0) {
        fprintf(fp, "Loihi_Intel,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->loihi_time, results->loihi_steps, results->loihi_error,
               results->loihi_accuracy * 100);
    }
    if (results->brainchips_time > 0) {
        fprintf(fp, "BrainChips,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->brainchips_time, results->brainchips_steps, results->brainchips_error,
               results->brainchips_accuracy * 100);
    }
    if (results->racetrack_time > 0) {
        fprintf(fp, "Racetrack_Parkin,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->racetrack_time, results->racetrack_steps, results->racetrack_error,
               results->racetrack_accuracy * 100);
    }
    if (results->pcm_time > 0) {
        fprintf(fp, "Phase_Change_Memory,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->pcm_time, results->pcm_steps, results->pcm_error,
               results->pcm_accuracy * 100);
    }
    if (results->lyric_time > 0) {
        fprintf(fp, "Lyric_MIT,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->lyric_time, results->lyric_steps, results->lyric_error,
               results->lyric_accuracy * 100);
    }
    if (results->hw_bayesian_time > 0) {
        fprintf(fp, "HW_Bayesian_Chandra,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->hw_bayesian_time, results->hw_bayesian_steps, results->hw_bayesian_error,
               results->hw_bayesian_accuracy * 100);
    }
    if (results->semantic_lexo_bs_time > 0) {
        fprintf(fp, "Semantic_Lexo_BS,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->semantic_lexo_bs_time, results->semantic_lexo_bs_steps, results->semantic_lexo_bs_error,
               results->semantic_lexo_bs_accuracy * 100);
    }
    if (results->kernelized_sps_bs_time > 0) {
        fprintf(fp, "Kernelized_SPS_BS,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->kernelized_sps_bs_time, results->kernelized_sps_bs_steps, results->kernelized_sps_bs_error,
               results->kernelized_sps_bs_accuracy * 100);
    }
    if (results->spiralizer_chord_time > 0) {
        fprintf(fp, "Spiralizer_Chord,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->spiralizer_chord_time, results->spiralizer_chord_steps, results->spiralizer_chord_error,
               results->spiralizer_chord_accuracy * 100);
    }
    if (results->lattice_waterfront_time > 0) {
        fprintf(fp, "Lattice_Waterfront,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->lattice_waterfront_time, results->lattice_waterfront_steps, results->lattice_waterfront_error,
               results->lattice_waterfront_accuracy * 100);
    }
    if (results->multiple_search_tree_time > 0) {
        fprintf(fp, "Multiple_Search_Tree,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->multiple_search_tree_time, results->multiple_search_tree_steps, results->multiple_search_tree_error,
               results->multiple_search_tree_accuracy * 100);
    }
    if (results->directed_diffusion_time > 0) {
        fprintf(fp, "Directed_Diffusion,%.6f,%zu,%.6e,%.6f,1.00,1\n",
               results->directed_diffusion_time, results->directed_diffusion_steps, results->directed_diffusion_error,
               results->directed_diffusion_accuracy * 100);
    }
    
    fclose(fp);
    return 0;
}
