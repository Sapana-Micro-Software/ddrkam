/*
 * Real-Time, Online, and Dynamic Numerical Methods Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "realtime_online.h"
#include "rk3.h"
#include "adams.h"
#include "euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================================
// Real-Time RK3 Implementation
// ============================================================================

int realtime_rk_init(RealtimeRKSolver* solver, size_t state_dim, double step_size,
                     DataCallback callback, void* callback_data) {
    if (!solver || state_dim == 0 || step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(RealtimeRKSolver));
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->current_time = 0.0;
    solver->buffer_size = 1000;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->buffer = (double*)malloc(solver->buffer_size * state_dim * sizeof(double));
    
    if (!solver->current_state || !solver->buffer) {
        realtime_rk_free(solver);
        return -1;
    }
    
    solver->callback = callback;
    solver->callback_data = callback_data;
    
    return 0;
}

void realtime_rk_free(RealtimeRKSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->buffer) free(solver->buffer);
    
    memset(solver, 0, sizeof(RealtimeRKSolver));
}

double realtime_rk_step(RealtimeRKSolver* solver, ODEFunction f, double t,
                       double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    clock_t start = clock();
    
    // Use standard RK3 step
    double t_new = rk3_step(f, t, y, solver->state_dim, h, params);
    
    clock_t end = clock();
    double step_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Update performance metrics
    solver->total_steps++;
    solver->avg_step_time = (solver->avg_step_time * (solver->total_steps - 1) + step_time) / solver->total_steps;
    
    // Store in buffer
    if (solver->buffer && solver->buffer_idx < solver->buffer_size) {
        size_t idx = solver->buffer_idx * solver->state_dim;
        memcpy(&solver->buffer[idx], y, solver->state_dim * sizeof(double));
        solver->buffer_idx++;
    }
    
    // Call callback for real-time processing
    if (solver->callback) {
        solver->callback(t_new, y, solver->state_dim, solver->callback_data);
    }
    
    solver->current_time = t_new;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t_new;
}

int realtime_rk_process_stream(RealtimeRKSolver* solver, ODEFunction f,
                               const double* stream_data, size_t stream_length,
                               void* params) {
    if (!solver || !f || !stream_data || stream_length == 0) {
        return -1;
    }
    
    double* y = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y) {
        return -1;
    }
    
    // Process streaming data
    for (size_t i = 0; i < stream_length; i++) {
        // Extract state from stream (assuming interleaved format)
        for (size_t j = 0; j < solver->state_dim; j++) {
            y[j] = stream_data[i * solver->state_dim + j];
        }
        
        // Process step
        realtime_rk_step(solver, f, solver->current_time, y, solver->step_size, params);
    }
    
    free(y);
    return 0;
}

// ============================================================================
// Online RK3 Implementation
// ============================================================================

int online_rk_init(OnlineRKSolver* solver, size_t state_dim, double initial_step_size,
                  double learning_rate) {
    if (!solver || state_dim == 0 || initial_step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(OnlineRKSolver));
    solver->state_dim = state_dim;
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    *solver->adaptive_step_size = initial_step_size;
    solver->learning_rate = learning_rate;
    solver->history_size = 100;
    solver->error_threshold = 1e-6;
    solver->min_step_size = initial_step_size * 0.1;
    solver->max_step_size = initial_step_size * 10.0;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->weight_history = (double*)malloc(solver->history_size * sizeof(double));
    
    if (!solver->current_state || !solver->adaptive_step_size || !solver->weight_history) {
        online_rk_free(solver);
        return -1;
    }
    
    return 0;
}

void online_rk_free(OnlineRKSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->weight_history) free(solver->weight_history);
    
    memset(solver, 0, sizeof(OnlineRKSolver));
}

double online_rk_step(OnlineRKSolver* solver, ODEFunction f, double t,
                     double* y, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    double h = *solver->adaptive_step_size;
    
    // Perform RK3 step
    double t_new = rk3_step(f, t, y, solver->state_dim, h, params);
    
    // Estimate error (simplified - would use embedded methods in practice)
    double error_estimate = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        double diff = fabs(y[i] - solver->current_state[i]);
        error_estimate += diff * diff;
    }
    error_estimate = sqrt(error_estimate);
    
    // Adapt step size online
    double new_step = online_rk_adapt_step_size(solver, error_estimate);
    *solver->adaptive_step_size = new_step;
    
    // Update history
    if (solver->weight_history) {
        solver->weight_history[solver->history_idx % solver->history_size] = new_step;
        solver->history_idx++;
    }
    
    solver->cumulative_error += error_estimate;
    solver->adaptation_count++;
    
    solver->current_time = t_new;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t_new;
}

double online_rk_adapt_step_size(OnlineRKSolver* solver, double error_estimate) {
    if (!solver || !solver->adaptive_step_size) {
        return 0.0;
    }
    
    double current_step = *solver->adaptive_step_size;
    double new_step = current_step;
    
    if (error_estimate > solver->error_threshold) {
        // Reduce step size
        new_step = current_step * (1.0 - solver->learning_rate);
    } else if (error_estimate < solver->error_threshold * 0.1) {
        // Increase step size
        new_step = current_step * (1.0 + solver->learning_rate);
    }
    
    // Clamp to bounds
    if (new_step < solver->min_step_size) {
        new_step = solver->min_step_size;
    }
    if (new_step > solver->max_step_size) {
        new_step = solver->max_step_size;
    }
    
    return new_step;
}

// ============================================================================
// Dynamic RK3 Implementation
// ============================================================================

int dynamic_rk_init(DynamicRKSolver* solver, size_t state_dim, double initial_step_size,
                   double adaptation_rate) {
    if (!solver || state_dim == 0 || initial_step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DynamicRKSolver));
    solver->state_dim = state_dim;
    solver->dynamic_step_size = (double*)malloc(sizeof(double));
    *solver->dynamic_step_size = initial_step_size;
    solver->adaptation_rate = adaptation_rate;
    solver->error_tolerance = 1e-6;
    solver->stability_tolerance = 1e-4;
    solver->adaptive_mode = 1;
    solver->history_size = 50;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->error_estimate = (double*)malloc(sizeof(double));
    solver->stability_estimate = (double*)malloc(sizeof(double));
    solver->parameter_history = (double**)malloc(solver->history_size * sizeof(double*));
    
    if (!solver->current_state || !solver->dynamic_step_size || 
        !solver->error_estimate || !solver->stability_estimate || !solver->parameter_history) {
        dynamic_rk_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        solver->parameter_history[i] = (double*)malloc(2 * sizeof(double)); // step_size, error
        if (!solver->parameter_history[i]) {
            dynamic_rk_free(solver);
            return -1;
        }
    }
    
    return 0;
}

void dynamic_rk_free(DynamicRKSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->dynamic_step_size) free(solver->dynamic_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    if (solver->stability_estimate) free(solver->stability_estimate);
    
    if (solver->parameter_history) {
        for (size_t i = 0; i < solver->history_size; i++) {
            if (solver->parameter_history[i]) {
                free(solver->parameter_history[i]);
            }
        }
        free(solver->parameter_history);
    }
    
    memset(solver, 0, sizeof(DynamicRKSolver));
}

double dynamic_rk_step(DynamicRKSolver* solver, ODEFunction f, double t,
                      double* y, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    double h = *solver->dynamic_step_size;
    
    // Perform RK3 step
    double t_new = rk3_step(f, t, y, solver->state_dim, h, params);
    
    // Estimate error and stability
    double error = 0.0;
    double stability = 0.0;
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        double diff = fabs(y[i] - solver->current_state[i]);
        error += diff * diff;
        stability += fabs(y[i]);
    }
    error = sqrt(error);
    stability = stability / solver->state_dim;
    
    *solver->error_estimate = error;
    *solver->stability_estimate = stability;
    
    // Dynamic adaptation
    if (solver->adaptive_mode) {
        dynamic_rk_adapt(solver, error, stability);
    }
    
    // Store in history
    if (solver->parameter_history) {
        size_t idx = solver->history_idx % solver->history_size;
        solver->parameter_history[idx][0] = *solver->dynamic_step_size;
        solver->parameter_history[idx][1] = error;
        solver->history_idx++;
    }
    
    solver->current_time = t_new;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t_new;
}

void dynamic_rk_adapt(DynamicRKSolver* solver, double error_estimate, double stability_estimate) {
    if (!solver || !solver->dynamic_step_size) {
        return;
    }
    
    double current_step = *solver->dynamic_step_size;
    double new_step = current_step;
    
    // Adapt based on error
    if (error_estimate > solver->error_tolerance) {
        new_step = current_step * (1.0 - solver->adaptation_rate);
    } else if (error_estimate < solver->error_tolerance * 0.1) {
        new_step = current_step * (1.0 + solver->adaptation_rate);
    }
    
    // Adapt based on stability
    if (stability_estimate > solver->stability_tolerance) {
        new_step *= 0.95; // Slightly reduce for stability
    }
    
    *solver->dynamic_step_size = new_step;
}

// ============================================================================
// Real-Time Adams Methods Implementation
// ============================================================================

int realtime_adams_init_new(RealtimeAdamsSolver* solver, size_t state_dim, double step_size,
                            DataCallback callback, void* callback_data) {
    if (!solver || state_dim == 0 || step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(RealtimeAdamsSolver));
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->history_size = 3;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->history_t = (double**)malloc(solver->history_size * sizeof(double*));
    solver->history_y = (double**)malloc(solver->history_size * sizeof(double*));
    solver->buffer = (double*)malloc(1000 * state_dim * sizeof(double));
    
    if (!solver->current_state || !solver->history_t || !solver->history_y || !solver->buffer) {
        realtime_adams_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        solver->history_t[i] = (double*)malloc(sizeof(double));
        solver->history_y[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->history_t[i] || !solver->history_y[i]) {
            realtime_adams_free(solver);
            return -1;
        }
    }
    
    solver->callback = callback;
    solver->callback_data = callback_data;
    
    return 0;
}

void realtime_adams_free_new(RealtimeAdamsSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->buffer) free(solver->buffer);
    
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
    
    memset(solver, 0, sizeof(RealtimeAdamsSolver));
}

double realtime_adams_step_new(RealtimeAdamsSolver* solver, ODEFunction f, double t,
                               double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    // Prepare history arrays (Adams expects flat array: y[0*n...1*n-1], y[1*n...2*n-1], y[2*n...3*n-1])
    double t_arr[3];
    double* y_flat = (double*)malloc(3 * solver->state_dim * sizeof(double));
    
    if (!y_flat) {
        return t;
    }
    
    for (size_t i = 0; i < 3 && i < solver->history_count; i++) {
        t_arr[i] = *solver->history_t[i];
        memcpy(&y_flat[i * solver->state_dim], solver->history_y[i], solver->state_dim * sizeof(double));
    }
    
    // Use Adams-Bashforth/Moulton
    double* y_pred = (double*)malloc(solver->state_dim * sizeof(double));
    double* y_corr = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (y_pred && y_corr) {
        adams_bashforth3(f, t_arr, y_flat, solver->state_dim, h, params, y_pred);
        adams_moulton3(f, t_arr, y_flat, solver->state_dim, h, params, y_pred, y_corr);
        memcpy(y, y_corr, solver->state_dim * sizeof(double));
    }
    
    if (y_flat) free(y_flat);
    
    if (y_pred) free(y_pred);
    if (y_corr) free(y_corr);
    
    // Update history
    if (solver->history_count < solver->history_size) {
        solver->history_count++;
    } else {
        // Shift history
        memmove(solver->history_t[0], solver->history_t[1], 2 * sizeof(double*));
        memmove(solver->history_y[0], solver->history_y[1], 2 * sizeof(double*));
    }
    
    *solver->history_t[solver->history_count - 1] = t + h;
    memcpy(solver->history_y[solver->history_count - 1], y, solver->state_dim * sizeof(double));
    
    // Callback
    if (solver->callback) {
        solver->callback(t + h, y, solver->state_dim, solver->callback_data);
    }
    
    solver->current_time = t + h;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t + h;
}

// ============================================================================
// Online Adams Methods Implementation
// ============================================================================

int online_adams_init(OnlineAdamsSolver* solver, size_t state_dim, double initial_step_size,
                     double learning_rate) {
    if (!solver || state_dim == 0 || initial_step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(OnlineAdamsSolver));
    solver->state_dim = state_dim;
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    *solver->adaptive_step_size = initial_step_size;
    solver->learning_rate = learning_rate;
    solver->history_size = 3;
    solver->error_threshold = 1e-6;
    solver->min_step_size = initial_step_size * 0.1;
    solver->max_step_size = initial_step_size * 10.0;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->history_t = (double**)malloc(solver->history_size * sizeof(double*));
    solver->history_y = (double**)malloc(solver->history_size * sizeof(double*));
    
    if (!solver->current_state || !solver->adaptive_step_size || 
        !solver->history_t || !solver->history_y) {
        online_adams_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        solver->history_t[i] = (double*)malloc(sizeof(double));
        solver->history_y[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->history_t[i] || !solver->history_y[i]) {
            online_adams_free(solver);
            return -1;
        }
    }
    
    return 0;
}

void online_adams_free(OnlineAdamsSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    
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
    
    memset(solver, 0, sizeof(OnlineAdamsSolver));
}

double online_adams_step(OnlineAdamsSolver* solver, ODEFunction f, double t,
                        double* y, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    double h = *solver->adaptive_step_size;
    
    // Prepare history (flat array format)
    double t_arr[3];
    double* y_flat = (double*)malloc(3 * solver->state_dim * sizeof(double));
    
    if (!y_flat) {
        return t;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        t_arr[i] = *solver->history_t[i];
        memcpy(&y_flat[i * solver->state_dim], solver->history_y[i], solver->state_dim * sizeof(double));
    }
    
    // Adams step
    double* y_pred = (double*)malloc(solver->state_dim * sizeof(double));
    double* y_corr = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (y_pred && y_corr) {
        adams_bashforth3(f, t_arr, y_flat, solver->state_dim, h, params, y_pred);
        adams_moulton3(f, t_arr, y_flat, solver->state_dim, h, params, y_pred, y_corr);
        memcpy(y, y_corr, solver->state_dim * sizeof(double));
    }
    
    if (y_flat) free(y_flat);
    
    if (y_pred) free(y_pred);
    if (y_corr) free(y_corr);
    
    // Estimate error and adapt
    double error = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        double diff = fabs(y[i] - solver->current_state[i]);
        error += diff * diff;
    }
    error = sqrt(error);
    
    // Adapt step size
    if (error > solver->error_threshold) {
        *solver->adaptive_step_size = h * (1.0 - solver->learning_rate);
    } else if (error < solver->error_threshold * 0.1) {
        *solver->adaptive_step_size = h * (1.0 + solver->learning_rate);
    }
    
    // Clamp
    if (*solver->adaptive_step_size < solver->min_step_size) {
        *solver->adaptive_step_size = solver->min_step_size;
    }
    if (*solver->adaptive_step_size > solver->max_step_size) {
        *solver->adaptive_step_size = solver->max_step_size;
    }
    
    // Update history
    memmove(solver->history_t[0], solver->history_t[1], 2 * sizeof(double*));
    memmove(solver->history_y[0], solver->history_y[1], 2 * sizeof(double*));
    *solver->history_t[2] = t + h;
    memcpy(solver->history_y[2], y, solver->state_dim * sizeof(double));
    
    solver->current_time = t + h;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t + h;
}

// ============================================================================
// Dynamic Adams Methods Implementation
// ============================================================================

int dynamic_adams_init(DynamicAdamsSolver* solver, size_t state_dim, double initial_step_size,
                      double adaptation_rate) {
    if (!solver || state_dim == 0 || initial_step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DynamicAdamsSolver));
    solver->state_dim = state_dim;
    solver->dynamic_step_size = (double*)malloc(sizeof(double));
    *solver->dynamic_step_size = initial_step_size;
    solver->adaptation_rate = adaptation_rate;
    solver->error_tolerance = 1e-6;
    solver->history_size = 3;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->error_estimate = (double*)malloc(sizeof(double));
    solver->history_t = (double**)malloc(solver->history_size * sizeof(double*));
    solver->history_y = (double**)malloc(solver->history_size * sizeof(double*));
    
    if (!solver->current_state || !solver->dynamic_step_size || 
        !solver->error_estimate || !solver->history_t || !solver->history_y) {
        dynamic_adams_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        solver->history_t[i] = (double*)malloc(sizeof(double));
        solver->history_y[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->history_t[i] || !solver->history_y[i]) {
            dynamic_adams_free(solver);
            return -1;
        }
    }
    
    return 0;
}

void dynamic_adams_free(DynamicAdamsSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->dynamic_step_size) free(solver->dynamic_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    
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
    
    memset(solver, 0, sizeof(DynamicAdamsSolver));
}

double dynamic_adams_step(DynamicAdamsSolver* solver, ODEFunction f, double t,
                         double* y, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    double h = *solver->dynamic_step_size;
    
    // Prepare history (flat array format)
    double t_arr[3];
    double* y_flat = (double*)malloc(3 * solver->state_dim * sizeof(double));
    
    if (!y_flat) {
        return t;
    }
    
    for (size_t i = 0; i < solver->history_size; i++) {
        t_arr[i] = *solver->history_t[i];
        memcpy(&y_flat[i * solver->state_dim], solver->history_y[i], solver->state_dim * sizeof(double));
    }
    
    // Adams step
    double* y_pred = (double*)malloc(solver->state_dim * sizeof(double));
    double* y_corr = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (y_pred && y_corr) {
        adams_bashforth3(f, t_arr, y_flat, solver->state_dim, h, params, y_pred);
        adams_moulton3(f, t_arr, y_flat, solver->state_dim, h, params, y_pred, y_corr);
        memcpy(y, y_corr, solver->state_dim * sizeof(double));
    }
    
    if (y_flat) free(y_flat);
    
    if (y_pred) free(y_pred);
    if (y_corr) free(y_corr);
    
    // Estimate error
    double error = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        double diff = fabs(y[i] - solver->current_state[i]);
        error += diff * diff;
    }
    error = sqrt(error);
    *solver->error_estimate = error;
    
    // Dynamic adaptation
    if (error > solver->error_tolerance) {
        *solver->dynamic_step_size = h * (1.0 - solver->adaptation_rate);
    } else if (error < solver->error_tolerance * 0.1) {
        *solver->dynamic_step_size = h * (1.0 + solver->adaptation_rate);
    }
    
    // Update history
    memmove(solver->history_t[0], solver->history_t[1], 2 * sizeof(double*));
    memmove(solver->history_y[0], solver->history_y[1], 2 * sizeof(double*));
    *solver->history_t[2] = t + h;
    memcpy(solver->history_y[2], y, solver->state_dim * sizeof(double));
    
    solver->current_time = t + h;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t + h;
}

// ============================================================================
// Real-Time Euler Implementation
// ============================================================================

int realtime_euler_init(RealtimeEulerSolver* solver, size_t state_dim, double step_size,
                       DataCallback callback, void* callback_data) {
    if (!solver || state_dim == 0 || step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(RealtimeEulerSolver));
    solver->state_dim = state_dim;
    solver->step_size = step_size;
    solver->buffer_size = 1000;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->buffer = (double*)malloc(solver->buffer_size * state_dim * sizeof(double));
    
    if (!solver->current_state || !solver->buffer) {
        realtime_euler_free(solver);
        return -1;
    }
    
    solver->callback = callback;
    solver->callback_data = callback_data;
    
    return 0;
}

void realtime_euler_free(RealtimeEulerSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->buffer) free(solver->buffer);
    
    memset(solver, 0, sizeof(RealtimeEulerSolver));
}

double realtime_euler_step(RealtimeEulerSolver* solver, ODEFunction f, double t,
                          double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    // Use standard Euler step
    double t_new = euler_step(f, t, y, solver->state_dim, h, params);
    
    // Store in buffer
    if (solver->buffer && solver->buffer_idx < solver->buffer_size) {
        size_t idx = solver->buffer_idx * solver->state_dim;
        memcpy(&solver->buffer[idx], y, solver->state_dim * sizeof(double));
        solver->buffer_idx++;
    }
    
    // Callback
    if (solver->callback) {
        solver->callback(t_new, y, solver->state_dim, solver->callback_data);
    }
    
    solver->current_time = t_new;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t_new;
}

// ============================================================================
// Online Euler Implementation
// ============================================================================

int online_euler_init(OnlineEulerSolver* solver, size_t state_dim, double initial_step_size,
                     double learning_rate) {
    if (!solver || state_dim == 0 || initial_step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(OnlineEulerSolver));
    solver->state_dim = state_dim;
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    *solver->adaptive_step_size = initial_step_size;
    solver->learning_rate = learning_rate;
    solver->error_threshold = 1e-4;
    solver->min_step_size = initial_step_size * 0.1;
    solver->max_step_size = initial_step_size * 10.0;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    
    if (!solver->current_state || !solver->adaptive_step_size) {
        online_euler_free(solver);
        return -1;
    }
    
    return 0;
}

void online_euler_free(OnlineEulerSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    
    memset(solver, 0, sizeof(OnlineEulerSolver));
}

double online_euler_step(OnlineEulerSolver* solver, ODEFunction f, double t,
                        double* y, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    double h = *solver->adaptive_step_size;
    
    // Euler step
    double t_new = euler_step(f, t, y, solver->state_dim, h, params);
    
    // Estimate error
    double error = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        double diff = fabs(y[i] - solver->current_state[i]);
        error += diff * diff;
    }
    error = sqrt(error);
    
    // Adapt step size
    if (error > solver->error_threshold) {
        *solver->adaptive_step_size = h * (1.0 - solver->learning_rate);
    } else if (error < solver->error_threshold * 0.1) {
        *solver->adaptive_step_size = h * (1.0 + solver->learning_rate);
    }
    
    // Clamp
    if (*solver->adaptive_step_size < solver->min_step_size) {
        *solver->adaptive_step_size = solver->min_step_size;
    }
    if (*solver->adaptive_step_size > solver->max_step_size) {
        *solver->adaptive_step_size = solver->max_step_size;
    }
    
    solver->current_time = t_new;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t_new;
}

// ============================================================================
// Dynamic Euler Implementation
// ============================================================================

int dynamic_euler_init(DynamicEulerSolver* solver, size_t state_dim, double initial_step_size,
                      double adaptation_rate) {
    if (!solver || state_dim == 0 || initial_step_size <= 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DynamicEulerSolver));
    solver->state_dim = state_dim;
    solver->dynamic_step_size = (double*)malloc(sizeof(double));
    *solver->dynamic_step_size = initial_step_size;
    solver->adaptation_rate = adaptation_rate;
    solver->error_tolerance = 1e-4;
    
    solver->current_state = (double*)malloc(state_dim * sizeof(double));
    solver->error_estimate = (double*)malloc(sizeof(double));
    
    if (!solver->current_state || !solver->dynamic_step_size || !solver->error_estimate) {
        dynamic_euler_free(solver);
        return -1;
    }
    
    return 0;
}

void dynamic_euler_free(DynamicEulerSolver* solver) {
    if (!solver) return;
    
    if (solver->current_state) free(solver->current_state);
    if (solver->dynamic_step_size) free(solver->dynamic_step_size);
    if (solver->error_estimate) free(solver->error_estimate);
    
    memset(solver, 0, sizeof(DynamicEulerSolver));
}

double dynamic_euler_step(DynamicEulerSolver* solver, ODEFunction f, double t,
                         double* y, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    double h = *solver->dynamic_step_size;
    
    // Euler step
    double t_new = euler_step(f, t, y, solver->state_dim, h, params);
    
    // Estimate error
    double error = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        double diff = fabs(y[i] - solver->current_state[i]);
        error += diff * diff;
    }
    error = sqrt(error);
    *solver->error_estimate = error;
    
    // Dynamic adaptation
    if (error > solver->error_tolerance) {
        *solver->dynamic_step_size = h * (1.0 - solver->adaptation_rate);
    } else if (error < solver->error_tolerance * 0.1) {
        *solver->dynamic_step_size = h * (1.0 + solver->adaptation_rate);
    }
    
    solver->current_time = t_new;
    memcpy(solver->current_state, y, solver->state_dim * sizeof(double));
    
    return t_new;
}
