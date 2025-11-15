/*
 * Parallel and Distributed Adams Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "parallel_adams.h"
#include "adams.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int parallel_adams_init(ParallelAdamsSolver* solver, size_t state_dim,
                        size_t num_workers, ParallelMode mode,
                        StackedConfig* stacked) {
    if (!solver || state_dim == 0 || num_workers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(ParallelAdamsSolver));
    solver->state_dim = state_dim;
    solver->num_workers = num_workers;
    solver->mode = mode;
    solver->rank = 0;
    solver->size = 1;
    solver->history_size = 3;
    
    // Allocate work ranges
    solver->work_ranges = (size_t*)malloc((num_workers + 1) * sizeof(size_t));
    if (!solver->work_ranges) {
        return -1;
    }
    
    size_t chunk_size = state_dim / num_workers;
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : state_dim;
    }
    
    // Allocate state arrays
    solver->local_state = (double*)malloc(state_dim * sizeof(double));
    solver->global_state = (double*)malloc(state_dim * sizeof(double));
    
    // Allocate history
    solver->history_t = (double**)malloc(solver->history_size * sizeof(double*));
    solver->history_y = (double**)malloc(solver->history_size * sizeof(double*));
    
    if (!solver->local_state || !solver->global_state || 
        !solver->history_t || !solver->history_y) {
        parallel_adams_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        solver->history_t[i] = (double*)malloc(sizeof(double));
        solver->history_y[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->history_t[i] || !solver->history_y[i]) {
            parallel_adams_free(solver);
            return -1;
        }
    }
    
    // Allocate threads for pthread mode
    if (mode == PARALLEL_PTHREAD) {
        solver->threads = (void**)malloc(num_workers * sizeof(pthread_t));
        solver->num_threads = num_workers;
        if (!solver->threads) {
            parallel_adams_free(solver);
            return -1;
        }
    }
    
    // Store stacked configuration
    if (stacked) {
        solver->stacked = (StackedConfig*)malloc(sizeof(StackedConfig));
        if (!solver->stacked) {
            parallel_adams_free(solver);
            return -1;
        }
        memcpy(solver->stacked, stacked, sizeof(StackedConfig));
    }
    
    return 0;
}

void parallel_adams_free(ParallelAdamsSolver* solver) {
    if (!solver) return;
    
    if (solver->work_ranges) free(solver->work_ranges);
    if (solver->local_state) free(solver->local_state);
    if (solver->global_state) free(solver->global_state);
    if (solver->threads) free(solver->threads);
    if (solver->stacked) free(solver->stacked);
    
    if (solver->history_t) {
        for (size_t i = 0; i < solver->history_size; i++) {
            if (solver->history_t[i]) free(solver->history_t[i]);
        }
        free(solver->history_t);
    }
    
    if (solver->history_y) {
        for (size_t i = 0; i < solver->history_size; i++) {
            if (solver->history_y[i]) free(solver->history_y[i]);
        }
        free(solver->history_y);
    }
    
    memset(solver, 0, sizeof(ParallelAdamsSolver));
}

double parallel_adams_step(ParallelAdamsSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    size_t n = solver->state_dim;
    
    // Prepare history arrays for Adams-Bashforth/Moulton
    double t_arr[3];
    double* y_arr[3];
    
    // Use stored history or current state
    for (size_t i = 0; i < 3; i++) {
        if (i < solver->history_size && solver->history_t[i]) {
            t_arr[i] = *solver->history_t[i];
            y_arr[i] = solver->history_y[i];
        } else {
            t_arr[i] = t - (2 - i) * h;
            y_arr[i] = y; // Fallback
        }
    }
    
    // Predictor step (Adams-Bashforth 3)
    double* y_pred = (double*)malloc(n * sizeof(double));
    if (!y_pred) {
        return t;
    }
    
    if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(solver->num_workers)
        for (size_t i = 0; i < n; i++) {
            y_pred[i] = y_arr[2][i] + h * (23.0 * (y_arr[2][i] - y_arr[1][i]) / h - 
                                           16.0 * (y_arr[1][i] - y_arr[0][i]) / h + 
                                           5.0 * (y_arr[0][i] - y_arr[0][i]) / h) / 12.0;
        }
#endif
    } else {
        // Simplified predictor (needs derivative computation)
        const double* y_arr_const[3] = {y_arr[0], y_arr[1], y_arr[2]};
        adams_bashforth3(f, t_arr, y_arr_const, n, h, params, y_pred);
    }
    
    // Corrector step (Adams-Moulton 3)
    double* y_corr = (double*)malloc(n * sizeof(double));
    if (!y_corr) {
        free(y_pred);
        return t;
    }
    
    const double* y_arr_const[3] = {y_arr[0], y_arr[1], y_arr[2]};
    adams_moulton3(f, t_arr, y_arr_const, n, h, params, y_pred, y_corr);
    
    // Update state (parallelized)
    if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(solver->num_workers)
        for (size_t i = 0; i < n; i++) {
            y[i] = y_corr[i];
        }
#endif
    } else {
        memcpy(y, y_corr, n * sizeof(double));
    }
    
    free(y_pred);
    free(y_corr);
    
    return t + h;
}

double stacked_adams_step(ParallelAdamsSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params) {
    if (!solver || !f || !y || !solver->stacked) {
        return parallel_adams_step(solver, f, t, y, h, params);
    }
    
    // First do standard Adams step
    double t_new = parallel_adams_step(solver, f, t, y, h, params);
    
    // Apply stacked/hierarchical refinement
    // This would use the hierarchical architecture from DDRK3/DDAM
    // For now, apply a simple correction based on stacked config
    if (solver->stacked->use_attention) {
        size_t n = solver->state_dim;
        double correction = solver->stacked->learning_rate * h;
        
        if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
            #pragma omp parallel for num_threads(solver->num_workers)
            for (size_t i = 0; i < n; i++) {
                y[i] += correction * 0.01 * sin(y[i]); // Simple attention-like correction
            }
#endif
        } else {
            for (size_t i = 0; i < n; i++) {
                y[i] += correction * 0.01 * sin(y[i]);
            }
        }
    }
    
    return t_new;
}

int concurrent_adams_execute(ParallelAdamsSolver* solvers[], size_t num_solvers,
                             ODEFunction f, double t, const double* y, double h,
                             void* params, double** results) {
    if (!solvers || num_solvers == 0 || !y || !results) {
        return -1;
    }
    
    for (size_t i = 0; i < num_solvers; i++) {
        results[i] = (double*)malloc(solvers[i]->state_dim * sizeof(double));
        if (!results[i]) {
            for (size_t j = 0; j < i; j++) {
                free(results[j]);
            }
            return -1;
        }
        memcpy(results[i], y, solvers[i]->state_dim * sizeof(double));
    }
    
    if (solvers[0]->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for
        for (size_t i = 0; i < num_solvers; i++) {
            if (solvers[i]->stacked) {
                stacked_adams_step(solvers[i], f, t, results[i], h, params);
            } else {
                parallel_adams_step(solvers[i], f, t, results[i], h, params);
            }
        }
#endif
    } else {
        for (size_t i = 0; i < num_solvers; i++) {
            if (solvers[i]->stacked) {
                stacked_adams_step(solvers[i], f, t, results[i], h, params);
            } else {
                parallel_adams_step(solvers[i], f, t, results[i], h, params);
            }
        }
    }
    
    return 0;
}

size_t parallel_adams_solve(ParallelAdamsSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            double h, void* params, double* t_out, double* y_out) {
    if (!solver || !f || h <= 0.0 || t_end <= t0 || !y0 || !t_out || !y_out) {
        return 0;
    }
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) {
        return 0;
    }
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    
    // Initialize history
    for (size_t i = 0; i < solver->history_size; i++) {
        *solver->history_t[i] = t0 + i * h;
        memcpy(solver->history_y[i], y0, solver->state_dim * sizeof(double));
    }
    
    double t_current = t0 + (solver->history_size - 1) * h;
    size_t step = solver->history_size - 1;
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    
    // Store initial conditions
    for (size_t i = 0; i < solver->history_size && i < max_steps; i++) {
        t_out[i] = *solver->history_t[i];
        for (size_t j = 0; j < solver->state_dim; j++) {
            y_out[i * solver->state_dim + j] = solver->history_y[i][j];
        }
    }
    
    // Integrate using parallel Adams
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        
        if (solver->stacked) {
            t_current = stacked_adams_step(solver, f, t_current, y_current, h_actual, params);
        } else {
            t_current = parallel_adams_step(solver, f, t_current, y_current, h_actual, params);
        }
        
        // Update history
        memmove(solver->history_t[0], solver->history_t[1], 2 * sizeof(double*));
        memmove(solver->history_y[0], solver->history_y[1], 2 * sizeof(double*));
        *solver->history_t[2] = t_current;
        memcpy(solver->history_y[2], y_current, solver->state_dim * sizeof(double));
        
        t_out[step] = t_current;
        for (size_t i = 0; i < solver->state_dim; i++) {
            y_out[step * solver->state_dim + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}
