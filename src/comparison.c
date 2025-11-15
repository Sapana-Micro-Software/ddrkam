/*
 * Method Comparison Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "comparison.h"
#include "euler.h"
#include "hierarchical_euler.h"
#include "rk3.h"
#include "hierarchical_rk.h"
#include "adams.h"
#include "parallel_rk.h"
#include "parallel_adams.h"
#include "parallel_euler.h"
#include "realtime_online.h"
#include "nonlinear_solver.h"
#include "distributed_solvers.h"
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
    printf("║ DDRK3       │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->ddrk3_time, results->ddrk3_steps, results->ddrk3_error, results->ddrk3_accuracy * 100);
    printf("║ AM          │ %8.6f │ %5zu │ %10.6e │ %17.6f%% ║\n",
           results->am_time, results->am_steps, results->am_error, results->am_accuracy * 100);
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
    if (results->distributed_datadriven_time > 0) {
        printf("║ Dist+DD Solver │ %8.6f │   N/A │       N/A │              N/A ║\n",
               results->distributed_datadriven_time);
    }
    if (results->online_datadriven_time > 0) {
        printf("║ Online+DD      │ %8.6f │   N/A │       N/A │              N/A ║\n",
               results->online_datadriven_time);
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
    if (results->ddrk3_time > 0 && results->ddrk3_time < min_time) {
        min_time = results->ddrk3_time;
        best_time = "DDRK3";
    }
    if (results->am_time > 0 && results->am_time < min_time) {
        min_time = results->am_time;
        best_time = "AM";
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
    if (results->ddrk3_accuracy > max_accuracy) {
        max_accuracy = results->ddrk3_accuracy;
        best_accuracy = "DDRK3";
    }
    if (results->am_accuracy > max_accuracy) {
        max_accuracy = results->am_accuracy;
        best_accuracy = "AM";
    }
    if (results->ddam_accuracy > max_accuracy) {
        max_accuracy = results->ddam_accuracy;
        best_accuracy = "DDAM";
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
    fprintf(fp, "DDRK3,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->ddrk3_time, results->ddrk3_steps, results->ddrk3_error, results->ddrk3_accuracy * 100);
    fprintf(fp, "AM,%.6f,%zu,%.6e,%.6f,1.00,1\n",
            results->am_time, results->am_steps, results->am_error, results->am_accuracy * 100);
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
    
    fclose(fp);
    return 0;
}
