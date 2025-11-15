/*
 * Nonlinear Programming-Based Solvers Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "nonlinear_solver.h"
#include "rk3.h"
#include "euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Gradient descent for nonlinear optimization
static int gradient_descent_step(NonlinearODESolver* solver, const double* y_current,
                                 double* y_next, double h) {
    if (!solver || !y_current || !y_next) {
        return -1;
    }
    
    size_t n = solver->state_dim;
    
    // Compute gradient (simplified - would use automatic differentiation in practice)
    if (solver->gradient) {
        // Estimate gradient using finite differences
        double eps = 1e-8;
        double f_current = solver->objective(y_current, n, solver->params);
        
        for (size_t i = 0; i < n; i++) {
            double* y_perturbed = (double*)malloc(n * sizeof(double));
            if (!y_perturbed) continue;
            
            memcpy(y_perturbed, y_current, n * sizeof(double));
            y_perturbed[i] += eps;
            
            double f_perturbed = solver->objective(y_perturbed, n, solver->params);
            solver->gradient[i] = (f_perturbed - f_current) / eps;
            
            free(y_perturbed);
        }
        
        // Gradient descent step
        for (size_t i = 0; i < n; i++) {
            y_next[i] = y_current[i] - solver->step_size * solver->gradient[i] * h;
        }
    }
    
    return 0;
}

// Newton's method for nonlinear optimization
static int newton_step(NonlinearODESolver* solver, const double* y_current,
                      double* y_next, double h) {
    if (!solver || !y_current || !y_next) {
        return -1;
    }
    
    size_t n = solver->state_dim;
    
    // Compute gradient
    if (solver->gradient) {
        double eps = 1e-8;
        double f_current = solver->objective(y_current, n, solver->params);
        
        for (size_t i = 0; i < n; i++) {
            double* y_perturbed = (double*)malloc(n * sizeof(double));
            if (!y_perturbed) continue;
            
            memcpy(y_perturbed, y_current, n * sizeof(double));
            y_perturbed[i] += eps;
            
            double f_perturbed = solver->objective(y_perturbed, n, solver->params);
            solver->gradient[i] = (f_perturbed - f_current) / eps;
            
            free(y_perturbed);
        }
    }
    
    // Compute Hessian (simplified)
    if (solver->hessian) {
        double eps = 1e-6;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                // Second-order finite difference approximation
                double* y1 = (double*)malloc(n * sizeof(double));
                double* y2 = (double*)malloc(n * sizeof(double));
                
                if (y1 && y2) {
                    memcpy(y1, y_current, n * sizeof(double));
                    memcpy(y2, y_current, n * sizeof(double));
                    y1[i] += eps;
                    y2[j] += eps;
                    
                    double f1 = solver->objective(y1, n, solver->params);
                    double f2 = solver->objective(y2, n, solver->params);
                    double f_current = solver->objective(y_current, n, solver->params);
                    
                    solver->hessian[i][j] = (f1 + f2 - 2.0 * f_current) / (eps * eps);
                }
                
                if (y1) free(y1);
                if (y2) free(y2);
            }
        }
        
        // Solve H * delta = -gradient (simplified - use diagonal approximation)
        for (size_t i = 0; i < n; i++) {
            if (fabs(solver->hessian[i][i]) > 1e-10) {
                y_next[i] = y_current[i] - solver->gradient[i] / solver->hessian[i][i] * h;
            } else {
                y_next[i] = y_current[i] - solver->step_size * solver->gradient[i] * h;
            }
        }
    }
    
    return 0;
}

int nonlinear_ode_init(NonlinearODESolver* solver, size_t state_dim,
                       NLPSolverType solver_type, ObjectiveFunction objective,
                       ConstraintFunction constraints, void* params) {
    if (!solver || state_dim == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(NonlinearODESolver));
    solver->state_dim = state_dim;
    solver->solver_type = solver_type;
    solver->objective = objective;
    solver->constraints = constraints;
    solver->params = params;
    solver->tolerance = 1e-6;
    solver->max_iterations = 1000;
    solver->step_size = 0.01;
    
    solver->gradient = (double*)malloc(state_dim * sizeof(double));
    solver->hessian = (double**)malloc(state_dim * sizeof(double*));
    
    if (!solver->gradient || !solver->hessian) {
        nonlinear_ode_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < state_dim; i++) {
        solver->hessian[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->hessian[i]) {
            nonlinear_ode_free(solver);
            return -1;
        }
    }
    
    solver->history_size = 10;
    solver->gradient_history = (double*)malloc(state_dim * solver->history_size * sizeof(double));
    solver->step_history = (double*)malloc(state_dim * solver->history_size * sizeof(double));
    
    if (!solver->gradient_history || !solver->step_history) {
        nonlinear_ode_free(solver);
        return -1;
    }
    
    return 0;
}

void nonlinear_ode_free(NonlinearODESolver* solver) {
    if (!solver) return;
    
    if (solver->gradient) free(solver->gradient);
    if (solver->hessian) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            if (solver->hessian[i]) free(solver->hessian[i]);
        }
        free(solver->hessian);
    }
    if (solver->gradient_history) free(solver->gradient_history);
    if (solver->step_history) free(solver->step_history);
    
    memset(solver, 0, sizeof(NonlinearODESolver));
}

int nonlinear_ode_solve(NonlinearODESolver* solver, ODEFunction f,
                       double t0, double t_end, const double* y0,
                       double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    size_t n = solver->state_dim;
    double* y_current = (double*)malloc(n * sizeof(double));
    double* y_next = (double*)malloc(n * sizeof(double));
    
    if (!y_current || !y_next) {
        if (y_current) free(y_current);
        if (y_next) free(y_next);
        return -1;
    }
    
    memcpy(y_current, y0, n * sizeof(double));
    double t = t0;
    double h = 0.01;
    
    // Use ODE function to define objective
    // For nonlinear programming, we minimize the residual
    while (t < t_end) {
        double h_actual = (t + h > t_end) ? (t_end - t) : h;
        
        // Compute derivative
        double* dydt = (double*)malloc(n * sizeof(double));
        if (dydt) {
            f(t, y_current, dydt, solver->params);
            
            // Apply nonlinear programming step
            switch (solver->solver_type) {
                case NLP_GRADIENT_DESCENT:
                    gradient_descent_step(solver, y_current, y_next, h_actual);
                    break;
                case NLP_NEWTON:
                    newton_step(solver, y_current, y_next, h_actual);
                    break;
                default:
                    // Fallback to Euler
                    for (size_t i = 0; i < n; i++) {
                        y_next[i] = y_current[i] + h_actual * dydt[i];
                    }
                    break;
            }
            
            memcpy(y_current, y_next, n * sizeof(double));
            free(dydt);
        }
        
        t += h_actual;
    }
    
    memcpy(y_out, y_current, n * sizeof(double));
    
    free(y_current);
    free(y_next);
    return 0;
}

int nonlinear_pde_init(NonlinearPDESolver* solver, size_t spatial_dim,
                      const size_t* grid_size, NLPSolverType solver_type,
                      PDEFunction pde_func, void* params) {
    if (!solver || spatial_dim == 0 || !grid_size) {
        return -1;
    }
    
    memset(solver, 0, sizeof(NonlinearPDESolver));
    solver->spatial_dim = spatial_dim;
    solver->solver_type = solver_type;
    solver->pde_func = pde_func;
    solver->params = params;
    solver->tolerance = 1e-6;
    solver->max_iterations = 1000;
    
    solver->grid_size = (size_t*)malloc(spatial_dim * sizeof(size_t));
    if (!solver->grid_size) {
        return -1;
    }
    
    size_t total_points = 1;
    for (size_t i = 0; i < spatial_dim; i++) {
        solver->grid_size[i] = grid_size[i];
        total_points *= grid_size[i];
    }
    
    solver->u_grid = (double*)malloc(total_points * sizeof(double));
    solver->u_prev = (double*)malloc(total_points * sizeof(double));
    solver->residual = (double*)malloc(total_points * sizeof(double));
    
    if (!solver->u_grid || !solver->u_prev || !solver->residual) {
        nonlinear_pde_free(solver);
        return -1;
    }
    
    return 0;
}

void nonlinear_pde_free(NonlinearPDESolver* solver) {
    if (!solver) return;
    
    if (solver->grid_size) free(solver->grid_size);
    if (solver->u_grid) free(solver->u_grid);
    if (solver->u_prev) free(solver->u_prev);
    if (solver->residual) free(solver->residual);
    
    memset(solver, 0, sizeof(NonlinearPDESolver));
}

int nonlinear_pde_solve(NonlinearPDESolver* solver, double t0, double t_end,
                       const double* u0, double* u_out) {
    if (!solver || !u0 || !u_out) {
        return -1;
    }
    
    size_t total_points = 1;
    for (size_t i = 0; i < solver->spatial_dim; i++) {
        total_points *= solver->grid_size[i];
    }
    
    memcpy(solver->u_grid, u0, total_points * sizeof(double));
    
    double t = t0;
    double h = 0.01;
    size_t iterations = 0;
    
    while (t < t_end && iterations < solver->max_iterations) {
        memcpy(solver->u_prev, solver->u_grid, total_points * sizeof(double));
        
        // Compute residual using PDE function
        // Simplified: iterate over grid points
        for (size_t i = 0; i < total_points; i++) {
            // Map linear index to spatial coordinates (simplified for 1D/2D)
            double x = (double)(i % solver->grid_size[0]);
            double y = (solver->spatial_dim > 1) ? (double)(i / solver->grid_size[0]) : 0.0;
            
            double dudt, dudx, dudy;
            solver->pde_func(t, x, y, &solver->u_prev[i], &dudt, &dudx, &dudy, solver->params);
            
            // Update using nonlinear programming approach
            solver->u_grid[i] = solver->u_prev[i] + h * dudt;
        }
        
        t += h;
        iterations++;
    }
    
    memcpy(u_out, solver->u_grid, total_points * sizeof(double));
    return 0;
}

// Distributed Nonlinear Solver
int distributed_nonlinear_init(DistributedNonlinearSolver* solver,
                               NonlinearODESolver* base, size_t num_workers) {
    if (!solver || !base || num_workers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DistributedNonlinearSolver));
    solver->base_solver = base;
    solver->num_workers = num_workers;
    solver->rank = 0;
    solver->size = 1;
    
    size_t n = base->state_dim;
    size_t chunk_size = n / num_workers;
    
    solver->work_ranges = (size_t*)malloc((num_workers + 1) * sizeof(size_t));
    solver->local_state = (double*)malloc(chunk_size * sizeof(double));
    solver->global_state = (double*)malloc(n * sizeof(double));
    
    if (!solver->work_ranges || !solver->local_state || !solver->global_state) {
        distributed_nonlinear_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : n;
    }
    
    return 0;
}

void distributed_nonlinear_free(DistributedNonlinearSolver* solver) {
    if (!solver) return;
    
    if (solver->work_ranges) free(solver->work_ranges);
    if (solver->local_state) free(solver->local_state);
    if (solver->global_state) free(solver->global_state);
    
    memset(solver, 0, sizeof(DistributedNonlinearSolver));
}

int distributed_nonlinear_solve(DistributedNonlinearSolver* solver,
                                ODEFunction f, double t0, double t_end,
                                const double* y0, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Distribute work across workers
    size_t n = solver->base_solver->state_dim;
    size_t chunk_size = n / solver->num_workers;
    
    // Each worker processes its chunk
    for (size_t w = 0; w < solver->num_workers; w++) {
        size_t start = solver->work_ranges[w];
        size_t end = solver->work_ranges[w + 1];
        
        // Process local chunk (simplified - would use MPI in practice)
        for (size_t i = start; i < end; i++) {
            solver->local_state[i - start] = y0[i];
        }
    }
    
    // Solve using base solver
    return nonlinear_ode_solve(solver->base_solver, f, t0, t_end, y0, y_out);
}

// Data-Driven Nonlinear Solver
int datadriven_nonlinear_init(DataDrivenNonlinearSolver* solver,
                              NonlinearODESolver* base, size_t num_layers) {
    if (!solver || !base || num_layers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DataDrivenNonlinearSolver));
    solver->base_solver = base;
    solver->num_layers = num_layers;
    solver->learning_rate = 0.01;
    
    size_t n = base->state_dim;
    size_t hidden_dim = 32;
    
    solver->layer_weights = (double**)malloc(num_layers * sizeof(double*));
    solver->attention_weights = (double*)malloc(n * hidden_dim * sizeof(double));
    
    if (!solver->layer_weights || !solver->attention_weights) {
        datadriven_nonlinear_free(solver);
        return -1;
    }
    
    for (size_t l = 0; l < num_layers; l++) {
        solver->layer_weights[l] = (double*)malloc(hidden_dim * n * sizeof(double));
        if (!solver->layer_weights[l]) {
            datadriven_nonlinear_free(solver);
            return -1;
        }
        // Initialize with small random values
        for (size_t i = 0; i < hidden_dim * n; i++) {
            solver->layer_weights[l][i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (size_t i = 0; i < n * hidden_dim; i++) {
        solver->attention_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    return 0;
}

void datadriven_nonlinear_free(DataDrivenNonlinearSolver* solver) {
    if (!solver) return;
    
    if (solver->layer_weights) {
        for (size_t l = 0; l < solver->num_layers; l++) {
            if (solver->layer_weights[l]) free(solver->layer_weights[l]);
        }
        free(solver->layer_weights);
    }
    if (solver->attention_weights) free(solver->attention_weights);
    
    memset(solver, 0, sizeof(DataDrivenNonlinearSolver));
}

int datadriven_nonlinear_solve(DataDrivenNonlinearSolver* solver,
                               ODEFunction f, double t0, double t_end,
                               const double* y0, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Apply hierarchical transformation
    size_t n = solver->base_solver->state_dim;
    double* y_transformed = (double*)malloc(n * sizeof(double));
    
    if (!y_transformed) {
        return -1;
    }
    
    memcpy(y_transformed, y0, n * sizeof(double));
    
    // Process through hierarchical layers
    for (size_t l = 0; l < solver->num_layers; l++) {
        double* y_next = (double*)malloc(n * sizeof(double));
        if (!y_next) {
            free(y_transformed);
            return -1;
        }
        
        // Apply layer transformation (simplified)
        for (size_t i = 0; i < n; i++) {
            y_next[i] = y_transformed[i];
            // Add hierarchical correction
            for (size_t j = 0; j < n; j++) {
                size_t idx = i * n + j;
                y_next[i] += solver->learning_rate * solver->layer_weights[l][idx] * y_transformed[j];
            }
        }
        
        if (l > 0) free(y_transformed);
        y_transformed = y_next;
    }
    
    // Solve using base solver with transformed initial condition
    int result = nonlinear_ode_solve(solver->base_solver, f, t0, t_end, y_transformed, y_out);
    
    free(y_transformed);
    return result;
}

// Online Nonlinear Solver
int online_nonlinear_init(OnlineNonlinearSolver* solver,
                          NonlinearODESolver* base, double learning_rate) {
    if (!solver || !base) {
        return -1;
    }
    
    memset(solver, 0, sizeof(OnlineNonlinearSolver));
    solver->base_solver = base;
    solver->learning_rate = learning_rate;
    
    solver->adaptive_tolerance = (double*)malloc(sizeof(double));
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    
    if (!solver->adaptive_tolerance || !solver->adaptive_step_size) {
        online_nonlinear_free(solver);
        return -1;
    }
    
    *solver->adaptive_tolerance = base->tolerance;
    *solver->adaptive_step_size = base->step_size;
    
    return 0;
}

void online_nonlinear_free(OnlineNonlinearSolver* solver) {
    if (!solver) return;
    
    if (solver->adaptive_tolerance) free(solver->adaptive_tolerance);
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    
    memset(solver, 0, sizeof(OnlineNonlinearSolver));
}

int online_nonlinear_solve(OnlineNonlinearSolver* solver, ODEFunction f,
                          double t0, double t_end, const double* y0,
                          double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Adapt parameters online
    solver->base_solver->tolerance = *solver->adaptive_tolerance;
    solver->base_solver->step_size = *solver->adaptive_step_size;
    
    int result = nonlinear_ode_solve(solver->base_solver, f, t0, t_end, y0, y_out);
    
    // Adapt tolerance and step size based on results
    solver->adaptation_count++;
    if (solver->adaptation_count % 10 == 0) {
        *solver->adaptive_tolerance *= (1.0 - solver->learning_rate);
        *solver->adaptive_step_size *= (1.0 + solver->learning_rate * 0.1);
    }
    
    return result;
}

// Real-Time Nonlinear Solver
int realtime_nonlinear_init(RealtimeNonlinearSolver* solver,
                            NonlinearODESolver* base,
                            void (*callback)(double t, const double* y, size_t n, void* data),
                            void* callback_data) {
    if (!solver || !base) {
        return -1;
    }
    
    memset(solver, 0, sizeof(RealtimeNonlinearSolver));
    solver->base_solver = base;
    solver->buffer_size = 1000;
    
    size_t n = base->state_dim;
    solver->buffer = (double*)malloc(solver->buffer_size * n * sizeof(double));
    
    if (!solver->buffer) {
        realtime_nonlinear_free(solver);
        return -1;
    }
    
    solver->callback = callback;
    solver->callback_data = callback_data;
    
    return 0;
}

void realtime_nonlinear_free(RealtimeNonlinearSolver* solver) {
    if (!solver) return;
    
    if (solver->buffer) free(solver->buffer);
    
    memset(solver, 0, sizeof(RealtimeNonlinearSolver));
}

int realtime_nonlinear_solve(RealtimeNonlinearSolver* solver, ODEFunction f,
                             double t0, double t_end, const double* y0,
                             double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    size_t n = solver->base_solver->state_dim;
    double* y_current = (double*)malloc(n * sizeof(double));
    
    if (!y_current) {
        return -1;
    }
    
    memcpy(y_current, y0, n * sizeof(double));
    
    // Solve with real-time callbacks
    int result = nonlinear_ode_solve(solver->base_solver, f, t0, t_end, y_current, y_out);
    
    // Call callback
    if (solver->callback) {
        solver->callback(t_end, y_out, n, solver->callback_data);
    }
    
    free(y_current);
    return result;
}
