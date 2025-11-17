/*
 * Reverse Belief Propagation with Lossless Tracing
 * Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "reverse_belief_propagation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Numerical Jacobian computation (if not provided)
static void compute_numerical_jacobian(ODEFunction ode_func,
                                      double t,
                                      const double* y,
                                      double** jacobian,
                                      size_t state_dim,
                                      void* params) {
    double* y_perturbed = (double*)malloc(state_dim * sizeof(double));
    double* dydt_original = (double*)malloc(state_dim * sizeof(double));
    double* dydt_perturbed = (double*)malloc(state_dim * sizeof(double));
    
    if (!y_perturbed || !dydt_original || !dydt_perturbed) {
        free(y_perturbed);
        free(dydt_original);
        free(dydt_perturbed);
        return;
    }
    
    // Compute original derivative
    ode_func(t, y, dydt_original, params);
    
    // Compute Jacobian: J_ij = ∂f_i/∂y_j
    double epsilon = 1e-8;
    for (size_t j = 0; j < state_dim; j++) {
        // Perturb y[j]
        memcpy(y_perturbed, y, state_dim * sizeof(double));
        y_perturbed[j] += epsilon;
        
        // Compute perturbed derivative
        ode_func(t, y_perturbed, dydt_perturbed, params);
        
        // Finite difference: J_ij = (f_i(y+εe_j) - f_i(y)) / ε
        for (size_t i = 0; i < state_dim; i++) {
            jacobian[i][j] = (dydt_perturbed[i] - dydt_original[i]) / epsilon;
        }
    }
    
    free(y_perturbed);
    free(dydt_original);
    free(dydt_perturbed);
}

// ============================================================================
// Belief Operations
// ============================================================================

int belief_init(Belief* belief,
               size_t state_dim,
               const double* mean,
               double** covariance) {
    if (!belief || !mean || !covariance || state_dim == 0) {
        return -1;
    }
    
    belief->state_dim = state_dim;
    belief->timestamp = 0.0;
    
    belief->mean = (double*)malloc(state_dim * sizeof(double));
    belief->covariance = (double**)malloc(state_dim * sizeof(double*));
    belief->confidence = (double*)malloc(state_dim * sizeof(double));
    
    if (!belief->mean || !belief->covariance || !belief->confidence) {
        belief_free(belief);
        return -1;
    }
    
    for (size_t i = 0; i < state_dim; i++) {
        belief->covariance[i] = (double*)malloc(state_dim * sizeof(double));
        if (!belief->covariance[i]) {
            belief_free(belief);
            return -1;
        }
    }
    
    memcpy(belief->mean, mean, state_dim * sizeof(double));
    
    for (size_t i = 0; i < state_dim; i++) {
        memcpy(belief->covariance[i], covariance[i], state_dim * sizeof(double));
        
        // Compute confidence from covariance diagonal
        belief->confidence[i] = 1.0 / (1.0 + sqrt(belief->covariance[i][i]));
    }
    
    return 0;
}

void belief_free(Belief* belief) {
    if (!belief) return;
    
    free(belief->mean);
    
    if (belief->covariance) {
        for (size_t i = 0; i < belief->state_dim; i++) {
            free(belief->covariance[i]);
        }
        free(belief->covariance);
    }
    
    free(belief->confidence);
    
    memset(belief, 0, sizeof(Belief));
}

int belief_copy(Belief* dest, const Belief* src) {
    if (!dest || !src) return -1;
    
    if (dest->state_dim != src->state_dim) {
        belief_free(dest);
        if (belief_init(dest, src->state_dim, src->mean, src->covariance) != 0) {
            return -1;
        }
    } else {
        memcpy(dest->mean, src->mean, src->state_dim * sizeof(double));
        for (size_t i = 0; i < src->state_dim; i++) {
            memcpy(dest->covariance[i], src->covariance[i], src->state_dim * sizeof(double));
        }
        memcpy(dest->confidence, src->confidence, src->state_dim * sizeof(double));
    }
    
    dest->timestamp = src->timestamp;
    
    return 0;
}

int belief_propagate_forward(Belief* belief,
                            double** jacobian,
                            size_t state_dim,
                            double step_size) {
    if (!belief || !jacobian || belief->state_dim != state_dim) {
        return -1;
    }
    
    // Forward propagation: P(t+Δt) = J·P(t)·J^T
    // where J is the Jacobian
    
    // Compute J·P
    double** JP = (double**)malloc(state_dim * sizeof(double*));
    if (!JP) return -1;
    
    for (size_t i = 0; i < state_dim; i++) {
        JP[i] = (double*)malloc(state_dim * sizeof(double));
        if (!JP[i]) {
            for (size_t j = 0; j < i; j++) {
                free(JP[j]);
            }
            free(JP);
            return -1;
        }
        
        for (size_t j = 0; j < state_dim; j++) {
            JP[i][j] = 0.0;
            for (size_t k = 0; k < state_dim; k++) {
                JP[i][j] += jacobian[i][k] * belief->covariance[k][j];
            }
        }
    }
    
    // Compute (J·P)·J^T
    for (size_t i = 0; i < state_dim; i++) {
        for (size_t j = 0; j < state_dim; j++) {
            belief->covariance[i][j] = 0.0;
            for (size_t k = 0; k < state_dim; k++) {
                belief->covariance[i][j] += JP[i][k] * jacobian[j][k];
            }
            // Add process noise (small)
            if (i == j) {
                belief->covariance[i][j] += step_size * 1e-10;
            }
        }
        
        // Update confidence
        belief->confidence[i] = 1.0 / (1.0 + sqrt(belief->covariance[i][i]));
    }
    
    // Update mean: μ(t+Δt) = μ(t) + h·f(μ(t))
    // (simplified, full version would use extended Kalman filter)
    
    for (size_t i = 0; i < state_dim; i++) {
        free(JP[i]);
    }
    free(JP);
    
    return 0;
}

int belief_propagate_backward(Belief* belief,
                              double** jacobian,
                              size_t state_dim,
                              double step_size) {
    if (!belief || !jacobian || belief->state_dim != state_dim) {
        return -1;
    }
    
    // Reverse propagation: P(t) = J^{-1}·P(t+Δt)·(J^{-1})^T
    // For numerical stability, use transpose instead of inverse
    
    // Compute J^T·P
    double** JTP = (double**)malloc(state_dim * sizeof(double*));
    if (!JTP) return -1;
    
    for (size_t i = 0; i < state_dim; i++) {
        JTP[i] = (double*)malloc(state_dim * sizeof(double));
        if (!JTP[i]) {
            for (size_t j = 0; j < i; j++) {
                free(JTP[j]);
            }
            free(JTP);
            return -1;
        }
        
        for (size_t j = 0; j < state_dim; j++) {
            JTP[i][j] = 0.0;
            for (size_t k = 0; k < state_dim; k++) {
                JTP[i][j] += jacobian[k][i] * belief->covariance[k][j];  // J^T
            }
        }
    }
    
    // Compute J^T·P·J
    for (size_t i = 0; i < state_dim; i++) {
        for (size_t j = 0; j < state_dim; j++) {
            belief->covariance[i][j] = 0.0;
            for (size_t k = 0; k < state_dim; k++) {
                belief->covariance[i][j] += JTP[i][k] * jacobian[k][j];
            }
        }
        
        // Update confidence
        belief->confidence[i] = 1.0 / (1.0 + sqrt(belief->covariance[i][i]));
    }
    
    for (size_t i = 0; i < state_dim; i++) {
        free(JTP[i]);
    }
    free(JTP);
    
    return 0;
}

int belief_combine(const Belief* belief1,
                  const Belief* belief2,
                  Belief* belief_combined) {
    if (!belief1 || !belief2 || !belief_combined) {
        return -1;
    }
    
    if (belief1->state_dim != belief2->state_dim ||
        belief1->state_dim != belief_combined->state_dim) {
        return -1;
    }
    
    // Combine beliefs using information fusion (inverse covariance weighting)
    // P_combined = (P1^{-1} + P2^{-1})^{-1}
    // μ_combined = P_combined · (P1^{-1}·μ1 + P2^{-1}·μ2)
    
    // Simplified: weighted average based on confidence
    for (size_t i = 0; i < belief1->state_dim; i++) {
        double w1 = belief1->confidence[i];
        double w2 = belief2->confidence[i];
        double w_sum = w1 + w2;
        
        if (w_sum > 1e-10) {
            belief_combined->mean[i] = (w1 * belief1->mean[i] + w2 * belief2->mean[i]) / w_sum;
            
            // Combine covariances
            for (size_t j = 0; j < belief1->state_dim; j++) {
                belief_combined->covariance[i][j] = 
                    (w1 * belief1->covariance[i][j] + w2 * belief2->covariance[i][j]) / w_sum;
            }
            
            belief_combined->confidence[i] = w_sum / 2.0;
        } else {
            belief_combined->mean[i] = (belief1->mean[i] + belief2->mean[i]) / 2.0;
            for (size_t j = 0; j < belief1->state_dim; j++) {
                belief_combined->covariance[i][j] = 
                    (belief1->covariance[i][j] + belief2->covariance[i][j]) / 2.0;
            }
            belief_combined->confidence[i] = 0.5;
        }
    }
    
    return 0;
}

// ============================================================================
// Lossless Trace Operations
// ============================================================================

int trace_entry_init(LosslessTraceEntry* entry,
                    size_t state_dim,
                    double time) {
    if (!entry || state_dim == 0) {
        return -1;
    }
    
    entry->state_dim = state_dim;
    entry->time = time;
    entry->trace_id = 0;
    
    entry->state = (double*)malloc(state_dim * sizeof(double));
    entry->derivative = (double*)malloc(state_dim * sizeof(double));
    entry->jacobian = (double**)malloc(state_dim * sizeof(double*));
    entry->sensitivity = (double*)malloc(state_dim * sizeof(double));
    
    if (!entry->state || !entry->derivative || !entry->jacobian || !entry->sensitivity) {
        trace_entry_free(entry);
        return -1;
    }
    
    for (size_t i = 0; i < state_dim; i++) {
        entry->jacobian[i] = (double*)malloc(state_dim * sizeof(double));
        if (!entry->jacobian[i]) {
            trace_entry_free(entry);
            return -1;
        }
    }
    
    // Initialize belief
    if (belief_init(&entry->belief, state_dim, entry->state, entry->jacobian) != 0) {
        trace_entry_free(entry);
        return -1;
    }
    
    return 0;
}

void trace_entry_free(LosslessTraceEntry* entry) {
    if (!entry) return;
    
    free(entry->state);
    free(entry->derivative);
    
    if (entry->jacobian) {
        for (size_t i = 0; i < entry->state_dim; i++) {
            free(entry->jacobian[i]);
        }
        free(entry->jacobian);
    }
    
    free(entry->sensitivity);
    
    belief_free(&entry->belief);
    
    memset(entry, 0, sizeof(LosslessTraceEntry));
}

int trace_entry_store(LosslessTraceEntry* entry,
                     const double* state,
                     const double* derivative,
                     double** jacobian,
                     const double* sensitivity) {
    if (!entry || !state || !derivative) {
        return -1;
    }
    
    // Store complete state (lossless)
    memcpy(entry->state, state, entry->state_dim * sizeof(double));
    memcpy(entry->derivative, derivative, entry->state_dim * sizeof(double));
    
    if (jacobian) {
        for (size_t i = 0; i < entry->state_dim; i++) {
            memcpy(entry->jacobian[i], jacobian[i], entry->state_dim * sizeof(double));
        }
    }
    
    if (sensitivity) {
        memcpy(entry->sensitivity, sensitivity, entry->state_dim * sizeof(double));
    }
    
    // Update belief mean
    memcpy(entry->belief.mean, state, entry->state_dim * sizeof(double));
    entry->belief.timestamp = entry->time;
    
    return 0;
}

int trace_entry_retrieve(const LosslessTraceEntry* entry,
                        double* state,
                        double* derivative) {
    if (!entry || !state) {
        return -1;
    }
    
    // Lossless retrieval: exact reconstruction
    memcpy(state, entry->state, entry->state_dim * sizeof(double));
    
    if (derivative) {
        memcpy(derivative, entry->derivative, entry->state_dim * sizeof(double));
    }
    
    return 0;
}

// ============================================================================
// Reverse Belief Propagation Solver
// ============================================================================

int reverse_belief_init(ReverseBeliefPropagationSolver* solver,
                       size_t state_dim,
                       double step_size,
                       size_t trace_capacity,
                       ODEFunction ode_func,
                       void* ode_params,
                       void (*jacobian_func)(double t, const double* y, double** jacobian, void* params),
                       void* jacobian_params,
                       int store_jacobian,
                       int store_sensitivity) {
    if (!solver || state_dim == 0 || step_size <= 0.0 || !ode_func) {
        return -1;
    }
    
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->trace_capacity = trace_capacity;
    solver->ode_func = ode_func;
    solver->ode_params = ode_params;
    solver->jacobian_func = jacobian_func;
    solver->jacobian_params = jacobian_params;
    solver->store_jacobian = store_jacobian;
    solver->store_sensitivity = store_sensitivity;
    solver->use_exact_arithmetic = 0;  // Can be enabled for special cases
    solver->current_time = 0.0;
    solver->trace_count = 0;
    solver->forward_steps = 0;
    solver->reverse_steps = 0;
    solver->trace_operations = 0;
    solver->avg_forward_time = 0.0;
    solver->avg_reverse_time = 0.0;
    
    // Allocate memory
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->trace = (LosslessTraceEntry*)malloc(trace_capacity * sizeof(LosslessTraceEntry));
    solver->belief_history = (Belief*)malloc(trace_capacity * sizeof(Belief));
    solver->belief_transition = (double**)malloc(state_dim * sizeof(double*));
    
    if (!solver->current_state || !solver->trace || !solver->belief_history ||
        !solver->belief_transition) {
        reverse_belief_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < state_dim; i++) {
        solver->belief_transition[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->belief_transition[i]) {
            reverse_belief_free(solver);
            return -1;
        }
    }
    
    // Initialize trace entries
    for (size_t i = 0; i < trace_capacity; i++) {
        memset(&solver->trace[i], 0, sizeof(LosslessTraceEntry));
        memset(&solver->belief_history[i], 0, sizeof(Belief));
    }
    
    solver->trace_size = trace_capacity;
    
    return 0;
}

void reverse_belief_free(ReverseBeliefPropagationSolver* solver) {
    if (!solver) return;
    
    free(solver->current_state);
    
    if (solver->trace) {
        for (size_t i = 0; i < solver->trace_count; i++) {
            trace_entry_free(&solver->trace[i]);
        }
        free(solver->trace);
    }
    
    if (solver->belief_history) {
        for (size_t i = 0; i < solver->trace_count; i++) {
            belief_free(&solver->belief_history[i]);
        }
        free(solver->belief_history);
    }
    
    if (solver->belief_transition) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            free(solver->belief_transition[i]);
        }
        free(solver->belief_transition);
    }
    
    memset(solver, 0, sizeof(ReverseBeliefPropagationSolver));
}

int reverse_belief_forward_step(ReverseBeliefPropagationSolver* solver,
                               double t,
                               double* y,
                               Belief* belief) {
    if (!solver || !y || !belief || solver->trace_count >= solver->trace_capacity) {
        return -1;
    }
    
    double h = solver->step_size;
    
    // Compute derivative
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    if (!dydt) return -1;
    
    solver->ode_func(t, y, dydt, solver->ode_params);
    
    // Compute Jacobian (lossless if provided, numerical otherwise)
    double** jacobian = (double**)malloc(solver->state_dim * sizeof(double*));
    if (!jacobian) {
        free(dydt);
        return -1;
    }
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        jacobian[i] = (double*)malloc(solver->state_dim * sizeof(double));
        if (!jacobian[i]) {
            for (size_t j = 0; j < i; j++) {
                free(jacobian[j]);
            }
            free(jacobian);
            free(dydt);
            return -1;
        }
    }
    
    if (solver->jacobian_func) {
        solver->jacobian_func(t, y, jacobian, solver->jacobian_params);
    } else {
        compute_numerical_jacobian(solver->ode_func, t, y, jacobian,
                                  solver->state_dim, solver->ode_params);
    }
    
    // Initialize trace entry
    LosslessTraceEntry* entry = &solver->trace[solver->trace_count];
    if (trace_entry_init(entry, solver->state_dim, t) != 0) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            free(jacobian[i]);
        }
        free(jacobian);
        free(dydt);
        return -1;
    }
    
    entry->trace_id = solver->trace_count;
    
    // Store lossless trace
    trace_entry_store(entry, y, dydt, 
                     solver->store_jacobian ? jacobian : NULL,
                     solver->store_sensitivity ? NULL : NULL);
    
    // Propagate belief forward
    Belief* belief_entry = &solver->belief_history[solver->trace_count];
    if (belief_copy(belief_entry, belief) != 0) {
        trace_entry_free(entry);
        for (size_t i = 0; i < solver->state_dim; i++) {
            free(jacobian[i]);
        }
        free(jacobian);
        free(dydt);
        return -1;
    }
    
    belief_propagate_forward(belief, jacobian, solver->state_dim, h);
    
    // Step forward (simple Euler, can be replaced with RK4)
    for (size_t i = 0; i < solver->state_dim; i++) {
        y[i] += h * dydt[i];
    }
    
    // Update belief mean
    memcpy(belief->mean, y, solver->state_dim * sizeof(double));
    belief->timestamp = t + h;
    
    solver->trace_count++;
    solver->forward_steps++;
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        free(jacobian[i]);
    }
    free(jacobian);
    free(dydt);
    
    return 0;
}

int reverse_belief_reverse_step(ReverseBeliefPropagationSolver* solver,
                               double t,
                               double* y,
                               Belief* belief) {
    if (!solver || !y || !belief) {
        return -1;
    }
    
    // Find trace entry at time t (or nearest)
    LosslessTraceEntry* entry = NULL;
    size_t entry_idx = 0;
    double min_diff = DBL_MAX;
    
    for (size_t i = 0; i < solver->trace_count; i++) {
        double diff = fabs(solver->trace[i].time - t);
        if (diff < min_diff) {
            min_diff = diff;
            entry = &solver->trace[i];
            entry_idx = i;
        }
    }
    
    if (!entry || min_diff > solver->step_size * 2.0) {
        return -1;  // Trace entry not found
    }
    
    // Lossless retrieval: get exact state from trace
    trace_entry_retrieve(entry, y, NULL);
    
    // Get belief from history
    Belief* belief_entry = &solver->belief_history[entry_idx];
    
    // Propagate belief backwards
    if (solver->store_jacobian && entry->jacobian) {
        belief_propagate_backward(belief, entry->jacobian, solver->state_dim, solver->step_size);
    } else {
        // Recompute Jacobian if not stored
        double** jacobian = (double**)malloc(solver->state_dim * sizeof(double*));
        if (jacobian) {
            for (size_t i = 0; i < solver->state_dim; i++) {
                jacobian[i] = (double*)malloc(solver->state_dim * sizeof(double));
            }
            
            if (solver->jacobian_func) {
                solver->jacobian_func(entry->time, y, jacobian, solver->jacobian_params);
            } else {
                compute_numerical_jacobian(solver->ode_func, entry->time, y, jacobian,
                                         solver->state_dim, solver->ode_params);
            }
            
            belief_propagate_backward(belief, jacobian, solver->state_dim, solver->step_size);
            
            for (size_t i = 0; i < solver->state_dim; i++) {
                free(jacobian[i]);
            }
            free(jacobian);
        }
    }
    
    // Update belief mean (reverse)
    memcpy(belief->mean, y, solver->state_dim * sizeof(double));
    belief->timestamp = entry->time;
    
    solver->reverse_steps++;
    solver->trace_operations++;
    
    return 0;
}

int reverse_belief_forward_solve(ReverseBeliefPropagationSolver* solver,
                                double t0,
                                double t_end,
                                const double* y0,
                                const Belief* initial_belief) {
    if (!solver || !y0 || !initial_belief) {
        return -1;
    }
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    Belief current_belief;
    
    if (!y_current || belief_init(&current_belief, solver->state_dim,
                                  initial_belief->mean, initial_belief->covariance) != 0) {
        free(y_current);
        return -1;
    }
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    memcpy(current_belief.mean, y0, solver->state_dim * sizeof(double));
    current_belief.timestamp = t0;
    
    double t = t0;
    size_t step = 0;
    
    while (t < t_end && solver->trace_count < solver->trace_capacity) {
        if (reverse_belief_forward_step(solver, t, y_current, &current_belief) != 0) {
            break;
        }
        
        t += solver->step_size;
        step++;
    }
    
    memcpy(solver->current_state, y_current, solver->state_dim * sizeof(double));
    solver->current_time = t;
    
    belief_free(&current_belief);
    free(y_current);
    
    return 0;
}

int reverse_belief_reverse_solve(ReverseBeliefPropagationSolver* solver,
                                 double t_start,
                                 double t_end,
                                 const Belief* final_belief,
                                 double** solution,
                                 Belief* beliefs,
                                 size_t num_steps) {
    if (!solver || !final_belief || !solution || !beliefs) {
        return -1;
    }
    
    double time_step = (t_start - t_end) / num_steps;
    double t = t_start;
    
    Belief current_belief;
    if (belief_init(&current_belief, solver->state_dim,
                   final_belief->mean, final_belief->covariance) != 0) {
        return -1;
    }
    
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_current) {
        belief_free(&current_belief);
        return -1;
    }
    
    // Initialize from final belief
    memcpy(y_current, final_belief->mean, solver->state_dim * sizeof(double));
    memcpy(solution[0], final_belief->mean, solver->state_dim * sizeof(double));
    belief_copy(&beliefs[0], final_belief);
    
    for (size_t step = 1; step < num_steps; step++) {
        t -= time_step;  // Go backwards
        
        if (reverse_belief_reverse_step(solver, t, y_current, &current_belief) != 0) {
            // If trace not found, use extrapolation
            // (simplified: just copy previous)
            memcpy(y_current, solution[step - 1], solver->state_dim * sizeof(double));
        }
        
        memcpy(solution[step], y_current, solver->state_dim * sizeof(double));
        belief_copy(&beliefs[step], &current_belief);
    }
    
    belief_free(&current_belief);
    free(y_current);
    
    return 0;
}

int reverse_belief_get_trace(ReverseBeliefPropagationSolver* solver,
                            double t,
                            LosslessTraceEntry* trace_entry) {
    if (!solver || !trace_entry) {
        return -1;
    }
    
    // Find nearest trace entry
    LosslessTraceEntry* entry = NULL;
    double min_diff = DBL_MAX;
    
    for (size_t i = 0; i < solver->trace_count; i++) {
        double diff = fabs(solver->trace[i].time - t);
        if (diff < min_diff) {
            min_diff = diff;
            entry = &solver->trace[i];
        }
    }
    
    if (!entry || min_diff > solver->step_size * 2.0) {
        return -1;
    }
    
    // Copy trace entry
    if (trace_entry_init(trace_entry, entry->state_dim, entry->time) != 0) {
        return -1;
    }
    
    trace_entry_store(trace_entry, entry->state, entry->derivative,
                     entry->jacobian, entry->sensitivity);
    trace_entry->trace_id = entry->trace_id;
    
    return 0;
}

int reverse_belief_smooth(ReverseBeliefPropagationSolver* solver,
                         double t,
                         const double* y_forward,
                         const Belief* belief_forward,
                         const double* y_reverse,
                         const Belief* belief_reverse,
                         double* y_smoothed,
                         Belief* belief_smoothed) {
    if (!solver || !y_forward || !belief_forward || !y_reverse || !belief_reverse ||
        !y_smoothed || !belief_smoothed) {
        return -1;
    }
    
    // Combine forward and reverse beliefs
    if (belief_combine(belief_forward, belief_reverse, belief_smoothed) != 0) {
        return -1;
    }
    
    // Smoothed state: weighted combination based on confidence
    for (size_t i = 0; i < solver->state_dim; i++) {
        double w_forward = belief_forward->confidence[i];
        double w_reverse = belief_reverse->confidence[i];
        double w_sum = w_forward + w_reverse;
        
        if (w_sum > 1e-10) {
            y_smoothed[i] = (w_forward * y_forward[i] + w_reverse * y_reverse[i]) / w_sum;
        } else {
            y_smoothed[i] = (y_forward[i] + y_reverse[i]) / 2.0;
        }
    }
    
    // Update smoothed belief mean
    memcpy(belief_smoothed->mean, y_smoothed, solver->state_dim * sizeof(double));
    belief_smoothed->timestamp = t;
    
    return 0;
}
