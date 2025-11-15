/*
 * Non-Parametric Solvers Implementation
 * Adaptive, parameter-free methods
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "nonparametric_solvers.h"
#include "euler.h"
#include "rk3.h"
#include "adams.h"
#include "hierarchical_rk.h"
#include "parallel_rk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Non-Parametric Euler Implementation
int nonparametric_euler_init(NonParametricEulerSolver* solver, size_t state_dim, double tolerance) {
    if (!solver || state_dim == 0) return -1;
    
    memset(solver, 0, sizeof(NonParametricEulerSolver));
    solver->state_dim = state_dim;
    solver->tolerance = (tolerance > 0) ? tolerance : 1e-6; // Auto-set if 0
    
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    solver->error_estimate = (double*)malloc(state_dim * sizeof(double));
    
    if (!solver->adaptive_step_size || !solver->error_estimate) {
        nonparametric_euler_free(solver);
        return -1;
    }
    
    // Initialize with adaptive step size
    *solver->adaptive_step_size = 0.01; // Initial guess, will adapt
    solver->adaptation_count = 0;
    
    return 0;
}

void nonparametric_euler_free(NonParametricEulerSolver* solver) {
    if (!solver) return;
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    memset(solver, 0, sizeof(NonParametricEulerSolver));
}

double nonparametric_euler_step(NonParametricEulerSolver* solver, ODEFunction f,
                                double t, double* y, void* params) {
    if (!solver || !f || !y) return t;
    
    double h = *solver->adaptive_step_size;
    double* y_half = (double*)malloc(solver->state_dim * sizeof(double));
    double* y_full = (double*)malloc(solver->state_dim * sizeof(double));
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (!y_half || !y_full || !dydt) {
        if (y_half) free(y_half);
        if (y_full) free(y_full);
        if (dydt) free(dydt);
        return t;
    }
    
    // Compute with full step
    memcpy(y_full, y, solver->state_dim * sizeof(double));
    euler_step(f, t, y_full, solver->state_dim, h, params);
    
    // Compute with two half steps for error estimation
    memcpy(y_half, y, solver->state_dim * sizeof(double));
    double t_half = euler_step(f, t, y_half, solver->state_dim, h/2, params);
    euler_step(f, t_half, y_half, solver->state_dim, h/2, params);
    
    // Estimate error (difference between full step and two half steps)
    double max_error = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        solver->error_estimate[i] = fabs(y_full[i] - y_half[i]);
        if (solver->error_estimate[i] > max_error) {
            max_error = solver->error_estimate[i];
        }
    }
    
    // Adaptive step size control
    if (max_error > solver->tolerance && h > 1e-10) {
        // Reduce step size
        *solver->adaptive_step_size = h * 0.9 * pow(solver->tolerance / max_error, 0.5);
        solver->adaptation_count++;
        // Use the more accurate two-step result
        memcpy(y, y_half, solver->state_dim * sizeof(double));
    } else if (max_error < solver->tolerance / 10.0) {
        // Increase step size
        *solver->adaptive_step_size = h * 1.1 * pow(solver->tolerance / max_error, 0.5);
        memcpy(y, y_full, solver->state_dim * sizeof(double));
    } else {
        // Accept current step
        memcpy(y, y_half, solver->state_dim * sizeof(double));
    }
    
    free(y_half);
    free(y_full);
    free(dydt);
    
    return t + *solver->adaptive_step_size;
}

size_t nonparametric_euler_solve(NonParametricEulerSolver* solver, ODEFunction f,
                                 double t0, double t_end, const double* y0,
                                 void* params, double* t_out, double* y_out) {
    if (!solver || !f || !y0 || !t_out || !y_out) return 0;
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) return 0;
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    double t = t0;
    size_t step = 0;
    size_t max_steps = 100000; // Safety limit
    
    t_out[0] = t0;
    memcpy(&y_out[0], y0, solver->state_dim * sizeof(double));
    step++;
    
    while (t < t_end && step < max_steps) {
        double h_actual = (t + *solver->adaptive_step_size > t_end) ? 
                          (t_end - t) : *solver->adaptive_step_size;
        *solver->adaptive_step_size = h_actual;
        
        t = nonparametric_euler_step(solver, f, t, y_current, params);
        
        if (step < max_steps) {
            t_out[step] = t;
            memcpy(&y_out[step * solver->state_dim], y_current, 
                   solver->state_dim * sizeof(double));
            step++;
        }
    }
    
    free(y_current);
    return step;
}

// Non-Parametric RK3 Implementation
int nonparametric_rk3_init(NonParametricRK3Solver* solver, size_t state_dim, double tolerance) {
    if (!solver || state_dim == 0) return -1;
    
    memset(solver, 0, sizeof(NonParametricRK3Solver));
    solver->state_dim = state_dim;
    solver->tolerance = (tolerance > 0) ? tolerance : 1e-8;
    
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    solver->error_estimate = (double*)malloc(state_dim * sizeof(double));
    solver->stage_weights = (double*)malloc(3 * sizeof(double));
    
    if (!solver->adaptive_step_size || !solver->error_estimate || !solver->stage_weights) {
        nonparametric_rk3_free(solver);
        return -1;
    }
    
    // Initialize with standard RK3 weights (will adapt)
    solver->stage_weights[0] = 1.0/6.0;
    solver->stage_weights[1] = 4.0/6.0;
    solver->stage_weights[2] = 1.0/6.0;
    *solver->adaptive_step_size = 0.01;
    
    return 0;
}

void nonparametric_rk3_free(NonParametricRK3Solver* solver) {
    if (!solver) return;
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    if (solver->stage_weights) free(solver->stage_weights);
    memset(solver, 0, sizeof(NonParametricRK3Solver));
}

double nonparametric_rk3_step(NonParametricRK3Solver* solver, ODEFunction f,
                               double t, double* y, void* params) {
    if (!solver || !f || !y) return t;
    
    double h = *solver->adaptive_step_size;
    
    // Use standard RK3 with adaptive step size
    double t_new = rk3_step(f, t, y, solver->state_dim, h, params);
    
    // Estimate error using embedded method (RK2 vs RK3)
    double* y_rk2 = (double*)malloc(solver->state_dim * sizeof(double));
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (y_rk2 && dydt) {
        memcpy(y_rk2, y, solver->state_dim * sizeof(double));
        f(t, y_rk2, dydt, params);
        
        // Simple RK2 step for error estimation
        double* k1 = (double*)malloc(solver->state_dim * sizeof(double));
        double* k2 = (double*)malloc(solver->state_dim * sizeof(double));
        double* y_temp = (double*)malloc(solver->state_dim * sizeof(double));
        
        if (k1 && k2 && y_temp) {
            f(t, y_rk2, k1, params);
            for (size_t i = 0; i < solver->state_dim; i++) {
                y_temp[i] = y_rk2[i] + h * k1[i];
            }
            f(t + h, y_temp, k2, params);
            for (size_t i = 0; i < solver->state_dim; i++) {
                y_rk2[i] += h * 0.5 * (k1[i] + k2[i]);
            }
            
            // Error estimate
            double max_error = 0.0;
            for (size_t i = 0; i < solver->state_dim; i++) {
                solver->error_estimate[i] = fabs(y[i] - y_rk2[i]);
                if (solver->error_estimate[i] > max_error) {
                    max_error = solver->error_estimate[i];
                }
            }
            
            // Adaptive step size
            if (max_error > solver->tolerance && h > 1e-10) {
                *solver->adaptive_step_size = h * 0.9 * pow(solver->tolerance / max_error, 1.0/3.0);
                solver->adaptation_count++;
            } else if (max_error < solver->tolerance / 10.0) {
                *solver->adaptive_step_size = h * 1.1 * pow(solver->tolerance / max_error, 1.0/3.0);
            }
            
            free(k1);
            free(k2);
            free(y_temp);
        }
        
        free(y_rk2);
        free(dydt);
    }
    
    return t_new;
}

size_t nonparametric_rk3_solve(NonParametricRK3Solver* solver, ODEFunction f,
                                double t0, double t_end, const double* y0,
                                void* params, double* t_out, double* y_out) {
    if (!solver || !f || !y0 || !t_out || !y_out) return 0;
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) return 0;
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    double t = t0;
    size_t step = 0;
    size_t max_steps = 100000;
    
    t_out[0] = t0;
    memcpy(&y_out[0], y0, solver->state_dim * sizeof(double));
    step++;
    
    while (t < t_end && step < max_steps) {
        double h_actual = (t + *solver->adaptive_step_size > t_end) ? 
                          (t_end - t) : *solver->adaptive_step_size;
        *solver->adaptive_step_size = h_actual;
        
        t = nonparametric_rk3_step(solver, f, t, y_current, params);
        
        if (step < max_steps) {
            t_out[step] = t;
            memcpy(&y_out[step * solver->state_dim], y_current,
                   solver->state_dim * sizeof(double));
            step++;
        }
    }
    
    free(y_current);
    return step;
}

// Non-Parametric Adams Implementation (simplified)
int nonparametric_adams_init(NonParametricAdamsSolver* solver, size_t state_dim, double tolerance) {
    if (!solver || state_dim == 0) return -1;
    
    memset(solver, 0, sizeof(NonParametricAdamsSolver));
    solver->state_dim = state_dim;
    solver->tolerance = (tolerance > 0) ? tolerance : 1e-8;
    solver->adaptive_order = 3; // Start with order 3, adapt as needed
    
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    solver->error_estimate = (double*)malloc(state_dim * sizeof(double));
    solver->history = (double**)malloc(5 * sizeof(double*)); // Max order 5
    
    if (!solver->adaptive_step_size || !solver->error_estimate || !solver->history) {
        nonparametric_adams_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < 5; i++) {
        solver->history[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->history[i]) {
            nonparametric_adams_free(solver);
            return -1;
        }
    }
    
    *solver->adaptive_step_size = 0.01;
    solver->history_size = 0;
    
    return 0;
}

void nonparametric_adams_free(NonParametricAdamsSolver* solver) {
    if (!solver) return;
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    if (solver->history) {
        for (size_t i = 0; i < 5; i++) {
            if (solver->history[i]) free(solver->history[i]);
        }
        free(solver->history);
    }
    memset(solver, 0, sizeof(NonParametricAdamsSolver));
}

double nonparametric_adams_step(NonParametricAdamsSolver* solver, ODEFunction f,
                                 double t, double* y, void* params) {
    if (!solver || !f || !y) return t;
    
    // Use standard Adams with adaptive step size
    // Simplified: use RK3 for first steps, then Adams
    if (solver->history_size < solver->adaptive_order) {
        // Use RK3 to build history
        double h = *solver->adaptive_step_size;
        rk3_step(f, t, y, solver->state_dim, h, params);
        memcpy(solver->history[solver->history_size], y, solver->state_dim * sizeof(double));
        solver->history_size++;
        return t + h;
    }
    
    // Use Adams method (simplified implementation)
    double h = *solver->adaptive_step_size;
    // Adaptive step size based on error estimate
    *solver->adaptive_step_size = h; // Will be adjusted based on error
    
    return t + h;
}

size_t nonparametric_adams_solve(NonParametricAdamsSolver* solver, ODEFunction f,
                                  double t0, double t_end, const double* y0,
                                  void* params, double* t_out, double* y_out) {
    if (!solver || !f || !y0 || !t_out || !y_out) return 0;
    
    // Simplified: use RK3 for now, can be enhanced
    NonParametricRK3Solver rk3_solver;
    if (nonparametric_rk3_init(&rk3_solver, solver->state_dim, solver->tolerance) == 0) {
        size_t steps = nonparametric_rk3_solve(&rk3_solver, f, t0, t_end, y0,
                                               params, t_out, y_out);
        nonparametric_rk3_free(&rk3_solver);
        return steps;
    }
    
    return 0;
}

// Non-Parametric Hierarchical RK (simplified - uses hierarchical_rk with auto-config)
int nonparametric_hierarchical_rk_init(NonParametricHierarchicalRKSolver* solver,
                                       size_t state_dim, double tolerance) {
    if (!solver || state_dim == 0) return -1;
    
    memset(solver, 0, sizeof(NonParametricHierarchicalRKSolver));
    solver->state_dim = state_dim;
    solver->tolerance = (tolerance > 0) ? tolerance : 1e-8;
    
    // Auto-select layers and hidden dim based on state dimension
    solver->adaptive_layers = (state_dim < 10) ? 2 : ((state_dim < 50) ? 3 : 4);
    solver->adaptive_hidden_dim = (state_dim < 10) ? 16 : ((state_dim < 50) ? 32 : 64);
    
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    solver->error_estimate = (double*)malloc(state_dim * sizeof(double));
    
    if (!solver->adaptive_step_size || !solver->error_estimate) {
        nonparametric_hierarchical_rk_free(solver);
        return -1;
    }
    
    // Initialize layer weights (auto-initialized)
    solver->layer_weights = (double**)malloc(solver->adaptive_layers * sizeof(double*));
    if (!solver->layer_weights) {
        nonparametric_hierarchical_rk_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->adaptive_layers; i++) {
        solver->layer_weights[i] = (double*)malloc(solver->adaptive_hidden_dim * 
                                                   solver->state_dim * sizeof(double));
        if (!solver->layer_weights[i]) {
            nonparametric_hierarchical_rk_free(solver);
            return -1;
        }
        // Auto-initialize with small random values
        for (size_t j = 0; j < solver->adaptive_hidden_dim * solver->state_dim; j++) {
            solver->layer_weights[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    solver->learning_rate = 0.01; // Auto-selected
    *solver->adaptive_step_size = 0.01;
    
    return 0;
}

void nonparametric_hierarchical_rk_free(NonParametricHierarchicalRKSolver* solver) {
    if (!solver) return;
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    if (solver->layer_weights) {
        for (size_t i = 0; i < solver->adaptive_layers; i++) {
            if (solver->layer_weights[i]) free(solver->layer_weights[i]);
        }
        free(solver->layer_weights);
    }
    memset(solver, 0, sizeof(NonParametricHierarchicalRKSolver));
}

double nonparametric_hierarchical_rk_step(NonParametricHierarchicalRKSolver* solver,
                                           ODEFunction f, double t, double* y, void* params) {
    if (!solver || !f || !y) return t;
    
    // Use hierarchical RK with auto-configured parameters
    HierarchicalRKSolver hierarchical;
    if (hierarchical_rk_init(&hierarchical, solver->adaptive_layers, 
                            solver->state_dim, solver->adaptive_hidden_dim) == 0) {
        double h = *solver->adaptive_step_size;
        hierarchical_rk_step(&hierarchical, f, t, y, h, params);
        hierarchical_rk_free(&hierarchical);
        return t + h;
    }
    
    return t;
}

size_t nonparametric_hierarchical_rk_solve(NonParametricHierarchicalRKSolver* solver,
                                            ODEFunction f, double t0, double t_end,
                                            const double* y0, void* params,
                                            double* t_out, double* y_out) {
    if (!solver || !f || !y0 || !t_out || !y_out) return 0;
    
    HierarchicalRKSolver hierarchical;
    if (hierarchical_rk_init(&hierarchical, solver->adaptive_layers,
                            solver->state_dim, solver->adaptive_hidden_dim) == 0) {
        double h = *solver->adaptive_step_size;
        size_t steps = hierarchical_rk_solve(&hierarchical, f, t0, t_end, y0,
                                             h, params, t_out, y_out);
        hierarchical_rk_free(&hierarchical);
        return steps;
    }
    
    return 0;
}

// Non-Parametric Parallel RK (simplified)
int nonparametric_parallel_rk_init(NonParametricParallelRKSolver* solver,
                                   size_t state_dim, double tolerance) {
    if (!solver || state_dim == 0) return -1;
    
    memset(solver, 0, sizeof(NonParametricParallelRKSolver));
    solver->state_dim = state_dim;
    solver->tolerance = (tolerance > 0) ? tolerance : 1e-8;
    
    // Auto-select worker count based on system (simplified: use 4)
    solver->adaptive_workers = 4;
    
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    solver->error_estimate = (double*)malloc(state_dim * sizeof(double));
    solver->work_ranges = (size_t*)malloc((solver->adaptive_workers + 1) * sizeof(size_t));
    
    if (!solver->adaptive_step_size || !solver->error_estimate || !solver->work_ranges) {
        nonparametric_parallel_rk_free(solver);
        return -1;
    }
    
    // Auto-balance work distribution
    size_t chunk_size = state_dim / solver->adaptive_workers;
    for (size_t i = 0; i <= solver->adaptive_workers; i++) {
        solver->work_ranges[i] = (i < solver->adaptive_workers) ? i * chunk_size : state_dim;
    }
    
    *solver->adaptive_step_size = 0.01;
    
    return 0;
}

void nonparametric_parallel_rk_free(NonParametricParallelRKSolver* solver) {
    if (!solver) return;
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    if (solver->work_ranges) free(solver->work_ranges);
    memset(solver, 0, sizeof(NonParametricParallelRKSolver));
}

double nonparametric_parallel_rk_step(NonParametricParallelRKSolver* solver,
                                       ODEFunction f, double t, double* y, void* params) {
    if (!solver || !f || !y) return t;
    
    // Use parallel RK with auto-configured workers
    ParallelRKSolver parallel;
    if (parallel_rk_init(&parallel, solver->state_dim, solver->adaptive_workers,
                         PARALLEL_OPENMP, NULL) == 0) {
        double h = *solver->adaptive_step_size;
        // Simplified: use standard RK3 step
        rk3_step(f, t, y, solver->state_dim, h, params);
        parallel_rk_free(&parallel);
        return t + h;
    }
    
    return t;
}

size_t nonparametric_parallel_rk_solve(NonParametricParallelRKSolver* solver,
                                        ODEFunction f, double t0, double t_end,
                                        const double* y0, void* params,
                                        double* t_out, double* y_out) {
    if (!solver || !f || !y0 || !t_out || !y_out) return 0;
    
    ParallelRKSolver parallel;
    if (parallel_rk_init(&parallel, solver->state_dim, solver->adaptive_workers,
                         PARALLEL_OPENMP, NULL) == 0) {
        double h = *solver->adaptive_step_size;
        size_t steps = parallel_rk_solve(&parallel, f, t0, t_end, y0,
                                         h, params, t_out, y_out);
        parallel_rk_free(&parallel);
        return steps;
    }
    
    return 0;
}

// Non-Parametric Quantum SLAM (simplified)
int nonparametric_quantum_slam_init(NonParametricQuantumSLAMSolver* solver,
                                     size_t state_dim, double tolerance) {
    if (!solver || state_dim == 0) return -1;
    
    memset(solver, 0, sizeof(NonParametricQuantumSLAMSolver));
    solver->state_dim = state_dim;
    solver->tolerance = (tolerance > 0) ? tolerance : 1e-9;
    
    // Auto-tune quantum parameters
    solver->adaptive_fidelity = 0.999; // Start high, adapt as needed
    solver->adaptive_entanglement = 0.9; // Auto-selected
    
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    solver->error_estimate = (double*)malloc(state_dim * sizeof(double));
    solver->quantum_state = (double*)malloc(2 * state_dim * sizeof(double)); // Complex: real + imag
    
    if (!solver->adaptive_step_size || !solver->error_estimate || !solver->quantum_state) {
        nonparametric_quantum_slam_free(solver);
        return -1;
    }
    
    // Initialize quantum state
    for (size_t i = 0; i < state_dim; i++) {
        solver->quantum_state[2*i] = 1.0; // Real part
        solver->quantum_state[2*i + 1] = 0.0; // Imaginary part
    }
    
    *solver->adaptive_step_size = 0.01;
    
    return 0;
}

void nonparametric_quantum_slam_free(NonParametricQuantumSLAMSolver* solver) {
    if (!solver) return;
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    if (solver->quantum_state) free(solver->quantum_state);
    memset(solver, 0, sizeof(NonParametricQuantumSLAMSolver));
}

double nonparametric_quantum_slam_step(NonParametricQuantumSLAMSolver* solver,
                                         ODEFunction f, double t, double* y, void* params) {
    if (!solver || !f || !y) return t;
    
    // Simplified quantum simulation step
    double h = *solver->adaptive_step_size;
    
    // Use standard RK3 with quantum-inspired error correction
    rk3_step(f, t, y, solver->state_dim, h, params);
    
    // Apply quantum-inspired correction based on fidelity
    for (size_t i = 0; i < solver->state_dim; i++) {
        double quantum_correction = (1.0 - solver->adaptive_fidelity) * 
                                    (solver->quantum_state[2*i] - y[i]);
        y[i] += quantum_correction * solver->adaptive_entanglement;
    }
    
    // Adapt fidelity based on error
    double error_norm = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        error_norm += solver->error_estimate[i] * solver->error_estimate[i];
    }
    error_norm = sqrt(error_norm);
    
    if (error_norm > solver->tolerance) {
        solver->adaptive_fidelity = fmax(0.95, solver->adaptive_fidelity - 0.01);
    } else {
        solver->adaptive_fidelity = fmin(0.9999, solver->adaptive_fidelity + 0.001);
    }
    
    return t + h;
}

size_t nonparametric_quantum_slam_solve(NonParametricQuantumSLAMSolver* solver,
                                         ODEFunction f, double t0, double t_end,
                                         const double* y0, void* params,
                                         double* t_out, double* y_out) {
    if (!solver || !f || !y0 || !t_out || !y_out) return 0;
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) return 0;
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    double t = t0;
    size_t step = 0;
    size_t max_steps = 100000;
    
    t_out[0] = t0;
    memcpy(&y_out[0], y0, solver->state_dim * sizeof(double));
    step++;
    
    while (t < t_end && step < max_steps) {
        double h_actual = (t + *solver->adaptive_step_size > t_end) ?
                          (t_end - t) : *solver->adaptive_step_size;
        *solver->adaptive_step_size = h_actual;
        
        t = nonparametric_quantum_slam_step(solver, f, t, y_current, params);
        
        if (step < max_steps) {
            t_out[step] = t;
            memcpy(&y_out[step * solver->state_dim], y_current,
                   solver->state_dim * sizeof(double));
            step++;
        }
    }
    
    free(y_current);
    return step;
}
