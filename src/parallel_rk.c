/*
 * Parallel and Distributed Runge-Kutta Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "parallel_rk.h"
#include "rk3.h"
#include "hierarchical_rk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Work structure for parallel execution
typedef struct {
    ParallelRKSolver* solver;
    ODEFunction f;
    double t;
    double* y;
    double h;
    void* params;
    size_t start_idx;
    size_t end_idx;
    double* k1;
    double* k2;
    double* k3;
    double* y_temp;
} ParallelWork;

// Thread worker function for pthread
static void* parallel_worker(void* arg) {
    ParallelWork* work = (ParallelWork*)arg;
    
    // Compute k1 for assigned range
    for (size_t i = work->start_idx; i < work->end_idx; i++) {
        // k1 computation is done globally, but we process our range
        work->y_temp[i] = work->y[i] + work->h * work->k1[i] / 2.0;
    }
    
    return NULL;
}

int parallel_rk_init(ParallelRKSolver* solver, size_t state_dim,
                     size_t num_workers, ParallelMode mode,
                     StackedConfig* stacked) {
    if (!solver || state_dim == 0 || num_workers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(ParallelRKSolver));
    solver->state_dim = state_dim;
    solver->num_workers = num_workers;
    solver->mode = mode;
    solver->rank = 0;
    solver->size = 1;
    
    // Allocate work ranges
    solver->work_ranges = (size_t*)malloc((num_workers + 1) * sizeof(size_t));
    if (!solver->work_ranges) {
        return -1;
    }
    
    // Distribute work evenly
    size_t chunk_size = state_dim / num_workers;
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : state_dim;
    }
    
    // Allocate state arrays
    solver->local_state = (double*)malloc(state_dim * sizeof(double));
    solver->global_state = (double*)malloc(state_dim * sizeof(double));
    
    if (!solver->local_state || !solver->global_state) {
        parallel_rk_free(solver);
        return -1;
    }
    
    // Allocate threads for pthread mode
    if (mode == PARALLEL_PTHREAD) {
        solver->threads = (void**)malloc(num_workers * sizeof(pthread_t));
        solver->num_threads = num_workers;
        if (!solver->threads) {
            parallel_rk_free(solver);
            return -1;
        }
    }
    
    // Store stacked configuration
    if (stacked) {
        solver->stacked = (StackedConfig*)malloc(sizeof(StackedConfig));
        if (!solver->stacked) {
            parallel_rk_free(solver);
            return -1;
        }
        memcpy(solver->stacked, stacked, sizeof(StackedConfig));
    }
    
    return 0;
}

void parallel_rk_free(ParallelRKSolver* solver) {
    if (!solver) return;
    
    if (solver->work_ranges) free(solver->work_ranges);
    if (solver->local_state) free(solver->local_state);
    if (solver->global_state) free(solver->global_state);
    if (solver->threads) free(solver->threads);
    if (solver->stacked) free(solver->stacked);
    
    memset(solver, 0, sizeof(ParallelRKSolver));
}

double parallel_rk_step(ParallelRKSolver* solver, ODEFunction f,
                        double t, double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    size_t n = solver->state_dim;
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    
    if (!k1 || !k2 || !k3 || !y_temp) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (k3) free(k3);
        if (y_temp) free(y_temp);
        return t;
    }
    
    // Compute k1 (can be parallelized)
    f(t, y, k1, params);
    
    // Parallel computation of k2
    if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(solver->num_workers)
        for (size_t i = 0; i < n; i++) {
            y_temp[i] = y[i] + h * k1[i] / 2.0;
        }
#endif
    } else if (solver->mode == PARALLEL_PTHREAD) {
        // Create work structures
        ParallelWork* works = (ParallelWork*)malloc(solver->num_workers * sizeof(ParallelWork));
        if (works) {
            for (size_t w = 0; w < solver->num_workers; w++) {
                works[w].solver = solver;
                works[w].f = f;
                works[w].t = t;
                works[w].y = y;
                works[w].h = h;
                works[w].params = params;
                works[w].start_idx = solver->work_ranges[w];
                works[w].end_idx = solver->work_ranges[w + 1];
                works[w].k1 = k1;
                works[w].k2 = k2;
                works[w].k3 = k3;
                works[w].y_temp = y_temp;
            }
            
            // Launch threads
            for (size_t w = 0; w < solver->num_workers; w++) {
                pthread_create((pthread_t*)&solver->threads[w], NULL, parallel_worker, &works[w]);
            }
            
            // Wait for completion
            for (size_t w = 0; w < solver->num_workers; w++) {
                pthread_join(*(pthread_t*)&solver->threads[w], NULL);
            }
            
            free(works);
        }
    } else {
        // Sequential fallback
        for (size_t i = 0; i < n; i++) {
            y_temp[i] = y[i] + h * k1[i] / 2.0;
        }
    }
    
    f(t + h/2.0, y_temp, k2, params);
    
    // Compute k3
    if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(solver->num_workers)
        for (size_t i = 0; i < n; i++) {
            y_temp[i] = y[i] - h * k1[i] + 2.0 * h * k2[i];
        }
#endif
    } else {
        for (size_t i = 0; i < n; i++) {
            y_temp[i] = y[i] - h * k1[i] + 2.0 * h * k2[i];
        }
    }
    
    f(t + h, y_temp, k3, params);
    
    // Final update (parallelized)
    if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(solver->num_workers)
        for (size_t i = 0; i < n; i++) {
            y[i] = y[i] + h * (k1[i] + 4.0 * k2[i] + k3[i]) / 6.0;
        }
#endif
    } else {
        for (size_t i = 0; i < n; i++) {
            y[i] = y[i] + h * (k1[i] + 4.0 * k2[i] + k3[i]) / 6.0;
        }
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(y_temp);
    
    return t + h;
}

double stacked_rk_step(ParallelRKSolver* solver, ODEFunction f,
                      double t, double* y, double h, void* params) {
    if (!solver || !f || !y || !solver->stacked) {
        return parallel_rk_step(solver, f, t, y, h, params);
    }
    
    // Use hierarchical RK for stacked execution
    HierarchicalRKSolver hierarchical;
    if (hierarchical_rk_init(&hierarchical, solver->stacked->num_layers,
                            solver->state_dim, solver->stacked->hidden_dim) == 0) {
        double result = hierarchical_rk_step(&hierarchical, f, t, y, h, params);
        hierarchical_rk_free(&hierarchical);
        return result;
    }
    
    // Fallback to standard parallel
    return parallel_rk_step(solver, f, t, y, h, params);
}

int concurrent_rk_execute(ParallelRKSolver* solvers[], size_t num_solvers,
                          ODEFunction f, double t, const double* y, double h,
                          void* params, double** results) {
    if (!solvers || num_solvers == 0 || !y || !results) {
        return -1;
    }
    
    // Allocate result arrays
    for (size_t i = 0; i < num_solvers; i++) {
        results[i] = (double*)malloc(solvers[i]->state_dim * sizeof(double));
        if (!results[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free(results[j]);
            }
            return -1;
        }
        memcpy(results[i], y, solvers[i]->state_dim * sizeof(double));
    }
    
    // Execute all solvers concurrently
    if (solvers[0]->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for
        for (size_t i = 0; i < num_solvers; i++) {
            if (solvers[i]->stacked) {
                stacked_rk_step(solvers[i], f, t, results[i], h, params);
            } else {
                parallel_rk_step(solvers[i], f, t, results[i], h, params);
            }
        }
#endif
    } else {
        // Sequential execution (fallback)
        for (size_t i = 0; i < num_solvers; i++) {
            if (solvers[i]->stacked) {
                stacked_rk_step(solvers[i], f, t, results[i], h, params);
            } else {
                parallel_rk_step(solvers[i], f, t, results[i], h, params);
            }
        }
    }
    
    return 0;
}

size_t parallel_rk_solve(ParallelRKSolver* solver, ODEFunction f,
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
    
    double t_current = t0;
    size_t step = 0;
    size_t max_steps = (size_t)((t_end - t0) / h) + 1;
    
    // Store initial condition
    t_out[step] = t_current;
    for (size_t i = 0; i < solver->state_dim; i++) {
        y_out[step * solver->state_dim + i] = y_current[i];
    }
    step++;
    
    // Integrate using parallel RK
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        
        if (solver->stacked) {
            t_current = stacked_rk_step(solver, f, t_current, y_current, h_actual, params);
        } else {
            t_current = parallel_rk_step(solver, f, t_current, y_current, h_actual, params);
        }
        
        t_out[step] = t_current;
        for (size_t i = 0; i < solver->state_dim; i++) {
            y_out[step * solver->state_dim + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}
