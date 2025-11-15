/*
 * Parallel and Distributed Euler Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "parallel_euler.h"
#include "euler.h"
#include "hierarchical_euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int parallel_euler_init(ParallelEulerSolver* solver, size_t state_dim,
                        size_t num_workers, ParallelMode mode,
                        StackedConfig* stacked) {
    if (!solver || state_dim == 0 || num_workers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(ParallelEulerSolver));
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
    
    size_t chunk_size = state_dim / num_workers;
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : state_dim;
    }
    
    // Allocate state arrays
    solver->local_state = (double*)malloc(state_dim * sizeof(double));
    solver->global_state = (double*)malloc(state_dim * sizeof(double));
    
    if (!solver->local_state || !solver->global_state) {
        parallel_euler_free(solver);
        return -1;
    }
    
    // Allocate threads for pthread mode
    if (mode == PARALLEL_PTHREAD) {
        solver->threads = (void**)malloc(num_workers * sizeof(pthread_t));
        solver->num_threads = num_workers;
        if (!solver->threads) {
            parallel_euler_free(solver);
            return -1;
        }
    }
    
    // Store stacked configuration
    if (stacked) {
        solver->stacked = (StackedConfig*)malloc(sizeof(StackedConfig));
        if (!solver->stacked) {
            parallel_euler_free(solver);
            return -1;
        }
        memcpy(solver->stacked, stacked, sizeof(StackedConfig));
    }
    
    return 0;
}

void parallel_euler_free(ParallelEulerSolver* solver) {
    if (!solver) return;
    
    if (solver->work_ranges) free(solver->work_ranges);
    if (solver->local_state) free(solver->local_state);
    if (solver->global_state) free(solver->global_state);
    if (solver->threads) free(solver->threads);
    if (solver->stacked) free(solver->stacked);
    
    memset(solver, 0, sizeof(ParallelEulerSolver));
}

double parallel_euler_step(ParallelEulerSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    size_t n = solver->state_dim;
    double* dydt = (double*)malloc(n * sizeof(double));
    
    if (!dydt) {
        return t;
    }
    
    // Compute derivative
    f(t, y, dydt, params);
    
    // Euler update (parallelized)
    if (solver->mode == PARALLEL_OPENMP) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(solver->num_workers)
        for (size_t i = 0; i < n; i++) {
            y[i] = y[i] + h * dydt[i];
        }
#endif
    } else {
        for (size_t i = 0; i < n; i++) {
            y[i] = y[i] + h * dydt[i];
        }
    }
    
    free(dydt);
    return t + h;
}

double stacked_euler_step(ParallelEulerSolver* solver, ODEFunction f,
                          double t, double* y, double h, void* params) {
    if (!solver || !f || !y || !solver->stacked) {
        return parallel_euler_step(solver, f, t, y, h, params);
    }
    
    // Use hierarchical Euler for stacked execution
    HierarchicalEulerSolver hierarchical;
    if (hierarchical_euler_init(&hierarchical, solver->stacked->num_layers,
                                solver->state_dim, solver->stacked->hidden_dim) == 0) {
        double result = hierarchical_euler_step(&hierarchical, f, t, y, h, params);
        hierarchical_euler_free(&hierarchical);
        return result;
    }
    
    // Fallback to standard parallel
    return parallel_euler_step(solver, f, t, y, h, params);
}

int concurrent_euler_execute(ParallelEulerSolver* solvers[], size_t num_solvers,
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
                stacked_euler_step(solvers[i], f, t, results[i], h, params);
            } else {
                parallel_euler_step(solvers[i], f, t, results[i], h, params);
            }
        }
#endif
    } else {
        for (size_t i = 0; i < num_solvers; i++) {
            if (solvers[i]->stacked) {
                stacked_euler_step(solvers[i], f, t, results[i], h, params);
            } else {
                parallel_euler_step(solvers[i], f, t, results[i], h, params);
            }
        }
    }
    
    return 0;
}

size_t parallel_euler_solve(ParallelEulerSolver* solver, ODEFunction f,
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
    
    // Integrate using parallel Euler
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        
        if (solver->stacked) {
            t_current = stacked_euler_step(solver, f, t_current, y_current, h_actual, params);
        } else {
            t_current = parallel_euler_step(solver, f, t_current, y_current, h_actual, params);
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
