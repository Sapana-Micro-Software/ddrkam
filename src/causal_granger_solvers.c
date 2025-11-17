/*
 * Causal and Granger Causality Real-Time ODE Solvers
 * Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "causal_granger_solvers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================================
// Causal RK4 Solver
// ============================================================================

int causal_rk4_init(CausalRK4Solver* solver,
                   size_t state_dim,
                   double step_size,
                   size_t history_size,
                   int strict_causality) {
    if (!solver || state_dim == 0 || step_size <= 0.0) {
        return -1;
    }
    
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->history_size = history_size;
    solver->strict_causality = strict_causality;
    solver->current_time = 0.0;
    solver->history_count = 0;
    solver->history_idx = 0;
    solver->buffer_idx = 0;
    solver->total_steps = 0;
    solver->avg_step_time = 0.0;
    
    // Allocate memory
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->k1 = (double*)malloc(state_dim * sizeof(double));
    solver->k2 = (double*)malloc(state_dim * sizeof(double));
    solver->k3 = (double*)malloc(state_dim * sizeof(double));
    solver->k4 = (double*)malloc(state_dim * sizeof(double));
    solver->y_temp = (double*)malloc(state_dim * sizeof(double));
    
    if (history_size > 0) {
        solver->state_history = (double**)malloc(history_size * sizeof(double*));
        solver->time_history = (double*)malloc(history_size * sizeof(double));
        
        if (!solver->state_history || !solver->time_history) {
            causal_rk4_free(solver);
            return -1;
        }
        
        for (size_t i = 0; i < history_size; i++) {
            solver->state_history[i] = (double*)malloc(state_dim * sizeof(double));
            if (!solver->state_history[i]) {
                causal_rk4_free(solver);
                return -1;
            }
        }
    } else {
        solver->state_history = NULL;
        solver->time_history = NULL;
    }
    
    solver->buffer_size = 10;  // Default buffer size
    solver->buffer = (double*)malloc(solver->buffer_size * state_dim * sizeof(double));
    
    if (!solver->current_state || !solver->k1 || !solver->k2 ||
        !solver->k3 || !solver->k4 || !solver->y_temp || !solver->buffer) {
        causal_rk4_free(solver);
        return -1;
    }
    
    return 0;
}

void causal_rk4_free(CausalRK4Solver* solver) {
    if (!solver) return;
    
    free(solver->current_state);
    free(solver->k1);
    free(solver->k2);
    free(solver->k3);
    free(solver->k4);
    free(solver->y_temp);
    free(solver->buffer);
    
    if (solver->state_history) {
        for (size_t i = 0; i < solver->history_size; i++) {
            free(solver->state_history[i]);
        }
        free(solver->state_history);
    }
    
    free(solver->time_history);
    
    memset(solver, 0, sizeof(CausalRK4Solver));
}

int causal_rk4_step(CausalRK4Solver* solver,
                   ODEFunction ode_func,
                   double t,
                   double* y,
                   void* params) {
    if (!solver || !ode_func || !y) {
        return -1;
    }
    
    double h = solver->step_size;
    
    // RK4 stages - all causal (only use current and past)
    // k1 = f(t, y)
    ode_func(t, y, solver->k1, params);
    
    // k2 = f(t + h/2, y + h*k1/2) - causal: uses current state
    for (size_t i = 0; i < solver->state_dim; i++) {
        solver->y_temp[i] = y[i] + 0.5 * h * solver->k1[i];
    }
    ode_func(t + 0.5 * h, solver->y_temp, solver->k2, params);
    
    // k3 = f(t + h/2, y + h*k2/2) - causal: uses current state
    for (size_t i = 0; i < solver->state_dim; i++) {
        solver->y_temp[i] = y[i] + 0.5 * h * solver->k2[i];
    }
    ode_func(t + 0.5 * h, solver->y_temp, solver->k3, params);
    
    // k4 = f(t + h, y + h*k3) - causal: uses current state
    for (size_t i = 0; i < solver->state_dim; i++) {
        solver->y_temp[i] = y[i] + h * solver->k3[i];
    }
    ode_func(t + h, solver->y_temp, solver->k4, params);
    
    // Update: y(t+h) = y(t) + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    // Strictly causal: only uses information at or before t
    for (size_t i = 0; i < solver->state_dim; i++) {
        y[i] += (h / 6.0) * (solver->k1[i] + 2.0 * solver->k2[i] +
                            2.0 * solver->k3[i] + solver->k4[i]);
    }
    
    // Update history (causal: only store past)
    if (solver->state_history && solver->history_size > 0) {
        memcpy(solver->state_history[solver->history_idx], y, 
               solver->state_dim * sizeof(double));
        solver->time_history[solver->history_idx] = t + h;
        
        solver->history_idx = (solver->history_idx + 1) % solver->history_size;
        if (solver->history_count < solver->history_size) {
            solver->history_count++;
        }
    }
    
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    solver->current_time = t + h;
    solver->total_steps++;
    
    return 0;
}

int causal_rk4_solve(CausalRK4Solver* solver,
                    ODEFunction ode_func,
                    double t0,
                    double t_end,
                    const double* y0,
                    void* params,
                    double** solution,
                    size_t num_steps) {
    if (!solver || !ode_func || !y0 || !solution) {
        return -1;
    }
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) return -1;
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    memcpy(solution[0], y0, solver->state_dim * sizeof(double));
    
    double time_step = (t_end - t0) / num_steps;
    double t = t0;
    
    for (size_t step = 1; step < num_steps; step++) {
        if (causal_rk4_step(solver, ode_func, t, y_current, params) != 0) {
            free(y_current);
            return -1;
        }
        
        memcpy(solution[step], y_current, solver->state_dim * sizeof(double));
        t += time_step;
    }
    
    free(y_current);
    return 0;
}

// ============================================================================
// Causal Adams Method Solver
// ============================================================================

int causal_adams_init(CausalAdamsSolver* solver,
                     size_t state_dim,
                     double step_size,
                     size_t adams_order,
                     size_t history_size,
                     int strict_causality) {
    if (!solver || state_dim == 0 || step_size <= 0.0 ||
        adams_order < 2 || adams_order > 4) {
        return -1;
    }
    
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->adams_order = adams_order;
    solver->history_size = history_size > adams_order ? history_size : adams_order + 1;
    solver->strict_causality = strict_causality;
    solver->current_time = 0.0;
    solver->history_count = 0;
    solver->history_idx = 0;
    solver->total_steps = 0;
    solver->avg_step_time = 0.0;
    
    // Allocate memory
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->state_history = (double**)malloc(solver->history_size * sizeof(double*));
    solver->derivative_history = (double**)malloc(solver->history_size * sizeof(double*));
    solver->time_history = (double*)malloc(solver->history_size * sizeof(double));
    
    if (!solver->current_state || !solver->state_history ||
        !solver->derivative_history || !solver->time_history) {
        causal_adams_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        solver->state_history[i] = (double*)malloc(state_dim * sizeof(double));
        solver->derivative_history[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->state_history[i] || !solver->derivative_history[i]) {
            causal_adams_free(solver);
            return -1;
        }
    }
    
    solver->buffer_size = 10;
    solver->buffer = (double*)malloc(solver->buffer_size * state_dim * sizeof(double));
    
    if (!solver->buffer) {
        causal_adams_free(solver);
        return -1;
    }
    
    return 0;
}

void causal_adams_free(CausalAdamsSolver* solver) {
    if (!solver) return;
    
    free(solver->current_state);
    free(solver->buffer);
    
    if (solver->state_history) {
        for (size_t i = 0; i < solver->history_size; i++) {
            free(solver->state_history[i]);
        }
        free(solver->state_history);
    }
    
    if (solver->derivative_history) {
        for (size_t i = 0; i < solver->history_size; i++) {
            free(solver->derivative_history[i]);
        }
        free(solver->derivative_history);
    }
    
    free(solver->time_history);
    
    memset(solver, 0, sizeof(CausalAdamsSolver));
}

int causal_adams_step(CausalAdamsSolver* solver,
                     ODEFunction ode_func,
                     double t,
                     double* y,
                     void* params) {
    if (!solver || !ode_func || !y) {
        return -1;
    }
    
    double h = solver->step_size;
    size_t order = solver->adams_order;
    
    // Compute current derivative (causal: only uses current state)
    double* f_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!f_current) return -1;
    
    ode_func(t, y, f_current, params);
    
    // Store in history
    memcpy(solver->state_history[solver->history_idx], y, 
           solver->state_dim * sizeof(double));
    memcpy(solver->derivative_history[solver->history_idx], f_current,
           solver->state_dim * sizeof(double));
    solver->time_history[solver->history_idx] = t;
    
    // Adams-Bashforth predictor (causal: only uses past derivatives)
    if (solver->history_count >= order) {
        // Adams-Bashforth coefficients for orders 2, 3, 4
        double coeffs[4][4] = {
            {0.0, 0.0, 0.0, 0.0},           // Order 1 (not used)
            {1.0, 1.0, 0.0, 0.0},           // Order 2: (3/2, -1/2)
            {23.0/12.0, -4.0/3.0, 5.0/12.0, 0.0},  // Order 3
            {55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0}  // Order 4
        };
        
        // Predict: y(t+h) = y(t) + h * Σ(β_i * f(t-i*h))
        // Only uses past information (causal)
        for (size_t i = 0; i < solver->state_dim; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < order; j++) {
                size_t hist_idx = (solver->history_idx - j + solver->history_size) % solver->history_size;
                sum += coeffs[order-1][j] * solver->derivative_history[hist_idx][i];
            }
            y[i] += h * sum;
        }
    } else {
        // Not enough history: use Euler step (causal)
        for (size_t i = 0; i < solver->state_dim; i++) {
            y[i] += h * f_current[i];
        }
    }
    
    free(f_current);
    
    // Update history index
    solver->history_idx = (solver->history_idx + 1) % solver->history_size;
    if (solver->history_count < solver->history_size) {
        solver->history_count++;
    }
    
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    solver->current_time = t + h;
    solver->total_steps++;
    
    return 0;
}

int causal_adams_solve(CausalAdamsSolver* solver,
                      ODEFunction ode_func,
                      double t0,
                      double t_end,
                      const double* y0,
                      void* params,
                      double** solution,
                      size_t num_steps) {
    if (!solver || !ode_func || !y0 || !solution) {
        return -1;
    }
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) return -1;
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    memcpy(solution[0], y0, solver->state_dim * sizeof(double));
    
    double time_step = (t_end - t0) / num_steps;
    double t = t0;
    
    for (size_t step = 1; step < num_steps; step++) {
        if (causal_adams_step(solver, ode_func, t, y_current, params) != 0) {
            free(y_current);
            return -1;
        }
        
        memcpy(solution[step], y_current, solver->state_dim * sizeof(double));
        t += time_step;
    }
    
    free(y_current);
    return 0;
}

// ============================================================================
// Granger Causality Solver
// ============================================================================

int granger_causality_init(GrangerCausalitySolver* solver,
                          size_t state_dim,
                          double step_size,
                          int base_method,
                          size_t adams_order,
                          size_t causality_window) {
    if (!solver || state_dim == 0 || step_size <= 0.0 || causality_window < 2) {
        return -1;
    }
    
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->base_method = base_method;
    solver->causality_window = causality_window;
    solver->current_time = 0.0;
    solver->history_count = 0;
    solver->history_idx = 0;
    solver->total_steps = 0;
    solver->causality_updates = 0;
    solver->avg_step_time = 0.0;
    
    // Allocate memory
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->causality_matrix = (double**)malloc(state_dim * sizeof(double*));
    solver->prediction_errors = (double**)malloc(state_dim * sizeof(double*));
    solver->variable_importance = (double*)malloc(state_dim * sizeof(double));
    solver->causal_dependencies = (int*)malloc(state_dim * sizeof(int));
    solver->adaptive_weights = (double*)malloc(state_dim * sizeof(double));
    
    solver->state_history = (double**)malloc(causality_window * sizeof(double*));
    solver->time_history = (double*)malloc(causality_window * sizeof(double));
    
    if (!solver->current_state || !solver->causality_matrix ||
        !solver->prediction_errors || !solver->variable_importance ||
        !solver->causal_dependencies || !solver->adaptive_weights ||
        !solver->state_history || !solver->time_history) {
        granger_causality_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < state_dim; i++) {
        solver->causality_matrix[i] = (double*)malloc(state_dim * sizeof(double));
        solver->prediction_errors[i] = (double*)malloc(state_dim * sizeof(double));
        
        if (!solver->causality_matrix[i] || !solver->prediction_errors[i]) {
            granger_causality_free(solver);
            return -1;
        }
        
        // Initialize
        for (size_t j = 0; j < state_dim; j++) {
            solver->causality_matrix[i][j] = 0.0;
            solver->prediction_errors[i][j] = 0.0;
        }
        solver->variable_importance[i] = 1.0 / state_dim;
        solver->causal_dependencies[i] = 0;
        solver->adaptive_weights[i] = 1.0;
    }
    
    for (size_t i = 0; i < causality_window; i++) {
        solver->state_history[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->state_history[i]) {
            granger_causality_free(solver);
            return -1;
        }
    }
    
    // Initialize base solver
    if (base_method == GRANGER_BASE_RK4) {
        CausalRK4Solver* rk4 = (CausalRK4Solver*)malloc(sizeof(CausalRK4Solver));
        if (!rk4 || causal_rk4_init(rk4, state_dim, step_size, causality_window, 1) != 0) {
            free(rk4);
            granger_causality_free(solver);
            return -1;
        }
        solver->base_solver = rk4;
    } else {
        CausalAdamsSolver* adams = (CausalAdamsSolver*)malloc(sizeof(CausalAdamsSolver));
        if (!adams || causal_adams_init(adams, state_dim, step_size, adams_order, causality_window, 1) != 0) {
            free(adams);
            granger_causality_free(solver);
            return -1;
        }
        solver->base_solver = adams;
    }
    
    return 0;
}

void granger_causality_free(GrangerCausalitySolver* solver) {
    if (!solver) return;
    
    free(solver->current_state);
    free(solver->variable_importance);
    free(solver->causal_dependencies);
    free(solver->adaptive_weights);
    free(solver->time_history);
    
    if (solver->causality_matrix) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            free(solver->causality_matrix[i]);
        }
        free(solver->causality_matrix);
    }
    
    if (solver->prediction_errors) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            free(solver->prediction_errors[i]);
        }
        free(solver->prediction_errors);
    }
    
    if (solver->state_history) {
        for (size_t i = 0; i < solver->causality_window; i++) {
            free(solver->state_history[i]);
        }
        free(solver->state_history);
    }
    
    // Free base solver
    if (solver->base_solver) {
        if (solver->base_method == GRANGER_BASE_RK4) {
            causal_rk4_free((CausalRK4Solver*)solver->base_solver);
        } else {
            causal_adams_free((CausalAdamsSolver*)solver->base_solver);
        }
        free(solver->base_solver);
    }
    
    memset(solver, 0, sizeof(GrangerCausalitySolver));
}

int granger_causality_update(GrangerCausalitySolver* solver) {
    if (!solver || solver->history_count < 2) {
        return -1;
    }
    
    // Granger causality: variable j causes variable i if including j improves prediction of i
    // Compute prediction errors with and without each variable
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        // Baseline error: predict i from its own past
        double baseline_error = 0.0;
        for (size_t t = 1; t < solver->history_count; t++) {
            size_t idx_t = (solver->history_idx - t + solver->causality_window) % solver->causality_window;
            size_t idx_t1 = (solver->history_idx - t + 1 + solver->causality_window) % solver->causality_window;
            
            double pred = solver->state_history[idx_t1][i];  // Simple: use previous value
            double actual = solver->state_history[idx_t][i];
            double diff = pred - actual;
            baseline_error += diff * diff;
        }
        baseline_error /= (solver->history_count - 1);
        
        // Test each variable j as a potential cause
        for (size_t j = 0; j < solver->state_dim; j++) {
            if (i == j) {
                solver->causality_matrix[i][j] = 1.0;  // Self-causality
                continue;
            }
            
            // Error when predicting i using both i's past and j's past
            double improved_error = 0.0;
            for (size_t t = 1; t < solver->history_count; t++) {
                size_t idx_t = (solver->history_idx - t + solver->causality_window) % solver->causality_window;
                size_t idx_t1 = (solver->history_idx - t + 1 + solver->causality_window) % solver->causality_window;
                
                // Linear prediction: y_i(t) ≈ α*y_i(t-1) + β*y_j(t-1)
                double alpha = 0.9;  // Weight on own past
                double beta = 0.1;   // Weight on j's past
                double pred = alpha * solver->state_history[idx_t1][i] +
                             beta * solver->state_history[idx_t1][j];
                double actual = solver->state_history[idx_t][i];
                double diff = pred - actual;
                improved_error += diff * diff;
            }
            improved_error /= (solver->history_count - 1);
            
            solver->prediction_errors[i][j] = improved_error;
            
            // Granger causality score: improvement in prediction
            // Higher score = stronger causality
            if (baseline_error > 1e-10) {
                solver->causality_matrix[i][j] = (baseline_error - improved_error) / baseline_error;
            } else {
                solver->causality_matrix[i][j] = 0.0;
            }
            
            // Threshold: only consider significant causality
            if (solver->causality_matrix[i][j] < 0.1) {
                solver->causality_matrix[i][j] = 0.0;
            }
        }
    }
    
    // Update variable importance (sum of incoming causalities)
    for (size_t i = 0; i < solver->state_dim; i++) {
        double importance = 0.0;
        for (size_t j = 0; j < solver->state_dim; j++) {
            importance += solver->causality_matrix[j][i];  // How much i causes others
        }
        solver->variable_importance[i] = importance;
    }
    
    // Normalize importance
    double sum = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        sum += solver->variable_importance[i];
    }
    if (sum > 1e-10) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            solver->variable_importance[i] /= sum;
        }
    }
    
    solver->causality_updates++;
    return 0;
}

int granger_causality_step(GrangerCausalitySolver* solver,
                          ODEFunction ode_func,
                          double t,
                          double* y,
                          void* params) {
    if (!solver || !ode_func || !y) {
        return -1;
    }
    
    // Store current state in history
    memcpy(solver->state_history[solver->history_idx], y,
           solver->state_dim * sizeof(double));
    solver->time_history[solver->history_idx] = t;
    
    solver->history_idx = (solver->history_idx + 1) % solver->causality_window;
    if (solver->history_count < solver->causality_window) {
        solver->history_count++;
    }
    
    // Update causality matrix periodically
    if (solver->history_count >= solver->causality_window &&
        solver->total_steps % 10 == 0) {  // Update every 10 steps
        granger_causality_update(solver);
    }
    
    // Use base solver (RK4 or Adams) with adaptive weights
    int result;
    if (solver->base_method == GRANGER_BASE_RK4) {
        // Apply adaptive weights based on causality
        double* y_weighted = (double*)malloc(solver->state_dim * sizeof(double));
        if (!y_weighted) return -1;
        
        for (size_t i = 0; i < solver->state_dim; i++) {
            y_weighted[i] = y[i] * solver->adaptive_weights[i];
        }
        
        result = causal_rk4_step((CausalRK4Solver*)solver->base_solver,
                                 ode_func, t, y_weighted, params);
        
        // Unweight result
        for (size_t i = 0; i < solver->state_dim; i++) {
            y[i] = y_weighted[i] / (solver->adaptive_weights[i] + 1e-10);
        }
        
        free(y_weighted);
    } else {
        result = causal_adams_step((CausalAdamsSolver*)solver->base_solver,
                                   ode_func, t, y, params);
    }
    
    if (result == 0) {
        memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
        solver->current_time = t + solver->step_size;
        solver->total_steps++;
    }
    
    return result;
}

int granger_causality_get_matrix(GrangerCausalitySolver* solver,
                                double** causality_matrix) {
    if (!solver || !causality_matrix) {
        return -1;
    }
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        memcpy(causality_matrix[i], solver->causality_matrix[i],
               solver->state_dim * sizeof(double));
    }
    
    return 0;
}

int granger_causality_get_importance(GrangerCausalitySolver* solver,
                                    double* importance) {
    if (!solver || !importance) {
        return -1;
    }
    
    memcpy(importance, solver->variable_importance,
           solver->state_dim * sizeof(double));
    
    return 0;
}

int granger_causality_solve(GrangerCausalitySolver* solver,
                           ODEFunction ode_func,
                           double t0,
                           double t_end,
                           const double* y0,
                           void* params,
                           double** solution,
                           size_t num_steps) {
    if (!solver || !ode_func || !y0 || !solution) {
        return -1;
    }
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) return -1;
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    memcpy(solution[0], y0, solver->state_dim * sizeof(double));
    
    double time_step = (t_end - t0) / num_steps;
    double t = t0;
    
    for (size_t step = 1; step < num_steps; step++) {
        if (granger_causality_step(solver, ode_func, t, y_current, params) != 0) {
            free(y_current);
            return -1;
        }
        
        memcpy(solution[step], y_current, solver->state_dim * sizeof(double));
        t += time_step;
    }
    
    free(y_current);
    return 0;
}
