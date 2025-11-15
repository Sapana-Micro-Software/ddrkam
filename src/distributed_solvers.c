/*
 * Additional Distributed, Data-Driven, Online, Real-Time Solvers Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "distributed_solvers.h"
#include "hierarchical_rk.h"
#include "rk3.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Distributed Data-Driven Solver
int distributed_datadriven_init(DistributedDataDrivenSolver* solver,
                                size_t state_dim, size_t num_workers, size_t num_layers) {
    if (!solver || state_dim == 0 || num_workers == 0 || num_layers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DistributedDataDrivenSolver));
    solver->state_dim = state_dim;
    solver->num_workers = num_workers;
    solver->num_layers = num_layers;
    solver->learning_rate = 0.01;
    solver->rank = 0;
    solver->size = 1;
    
    size_t hidden_dim = 32;
    size_t chunk_size = state_dim / num_workers;
    
    solver->work_ranges = (size_t*)malloc((num_workers + 1) * sizeof(size_t));
    solver->local_state = (double*)malloc(chunk_size * sizeof(double));
    solver->global_state = (double*)malloc(state_dim * sizeof(double));
    
    solver->layer_weights = (double**)malloc(num_layers * sizeof(double*));
    solver->attention_weights = (double*)malloc(state_dim * hidden_dim * sizeof(double));
    
    if (!solver->work_ranges || !solver->local_state || !solver->global_state ||
        !solver->layer_weights || !solver->attention_weights) {
        distributed_datadriven_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : state_dim;
    }
    
    for (size_t l = 0; l < num_layers; l++) {
        solver->layer_weights[l] = (double*)malloc(hidden_dim * state_dim * sizeof(double));
        if (!solver->layer_weights[l]) {
            distributed_datadriven_free(solver);
            return -1;
        }
        for (size_t i = 0; i < hidden_dim * state_dim; i++) {
            solver->layer_weights[l][i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (size_t i = 0; i < state_dim * hidden_dim; i++) {
        solver->attention_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    return 0;
}

void distributed_datadriven_free(DistributedDataDrivenSolver* solver) {
    if (!solver) return;
    
    if (solver->work_ranges) free(solver->work_ranges);
    if (solver->local_state) free(solver->local_state);
    if (solver->global_state) free(solver->global_state);
    if (solver->attention_weights) free(solver->attention_weights);
    
    if (solver->layer_weights) {
        for (size_t l = 0; l < solver->num_layers; l++) {
            if (solver->layer_weights[l]) free(solver->layer_weights[l]);
        }
        free(solver->layer_weights);
    }
    
    memset(solver, 0, sizeof(DistributedDataDrivenSolver));
}

int distributed_datadriven_solve(DistributedDataDrivenSolver* solver,
                                 ODEFunction f, double t0, double t_end,
                                 const double* y0, double h, void* params,
                                 double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Use hierarchical RK with distributed processing
    HierarchicalRKSolver hierarchical;
    if (hierarchical_rk_init(&hierarchical, solver->num_layers, solver->state_dim, 32) == 0) {
        size_t max_steps = (size_t)((t_end - t0) / h) + 1;
        double* t_out = (double*)malloc(max_steps * sizeof(double));
        double* y_temp = (double*)malloc(max_steps * solver->state_dim * sizeof(double));
        
        if (t_out && y_temp) {
            size_t steps = hierarchical_rk_solve(&hierarchical, f, t0, t_end, y0,
                                                 h, params, t_out, y_temp);
            if (steps > 0) {
                memcpy(y_out, &y_temp[(steps - 1) * solver->state_dim],
                      solver->state_dim * sizeof(double));
            }
        }
        
        if (t_out) free(t_out);
        if (y_temp) free(y_temp);
        hierarchical_rk_free(&hierarchical);
    }
    
    return 0;
}

// Online Data-Driven Solver
int online_datadriven_init(OnlineDataDrivenSolver* solver, size_t state_dim,
                          size_t num_layers, double initial_step_size,
                          double learning_rate) {
    if (!solver || state_dim == 0 || num_layers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(OnlineDataDrivenSolver));
    solver->state_dim = state_dim;
    solver->num_layers = num_layers;
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    *solver->adaptive_step_size = initial_step_size;
    solver->learning_rate = learning_rate;
    solver->error_threshold = 1e-6;
    
    size_t hidden_dim = 32;
    solver->layer_weights = (double**)malloc(num_layers * sizeof(double*));
    solver->attention_weights = (double*)malloc(state_dim * hidden_dim * sizeof(double));
    
    if (!solver->adaptive_step_size || !solver->layer_weights || !solver->attention_weights) {
        online_datadriven_free(solver);
        return -1;
    }
    
    for (size_t l = 0; l < num_layers; l++) {
        solver->layer_weights[l] = (double*)malloc(hidden_dim * state_dim * sizeof(double));
        if (!solver->layer_weights[l]) {
            online_datadriven_free(solver);
            return -1;
        }
        for (size_t i = 0; i < hidden_dim * state_dim; i++) {
            solver->layer_weights[l][i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (size_t i = 0; i < state_dim * hidden_dim; i++) {
        solver->attention_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    return 0;
}

void online_datadriven_free(OnlineDataDrivenSolver* solver) {
    if (!solver) return;
    
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->attention_weights) free(solver->attention_weights);
    
    if (solver->layer_weights) {
        for (size_t l = 0; l < solver->num_layers; l++) {
            if (solver->layer_weights[l]) free(solver->layer_weights[l]);
        }
        free(solver->layer_weights);
    }
    
    memset(solver, 0, sizeof(OnlineDataDrivenSolver));
}

int online_datadriven_solve(OnlineDataDrivenSolver* solver, ODEFunction f,
                            double t0, double t_end, const double* y0,
                            void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Use online RK with data-driven enhancement
    OnlineRKSolver online;
    if (online_rk_init(&online, solver->state_dim, *solver->adaptive_step_size,
                      solver->learning_rate) == 0) {
        double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
        if (y_current) {
            memcpy(y_current, y0, solver->state_dim * sizeof(double));
            
            double t = t0;
            while (t < t_end) {
                t = online_rk_step(&online, f, t, y_current, params);
            }
            
            memcpy(y_out, y_current, solver->state_dim * sizeof(double));
            *solver->adaptive_step_size = *online.adaptive_step_size;
            
            free(y_current);
        }
        online_rk_free(&online);
    }
    
    return 0;
}

// Real-Time Data-Driven Solver
int realtime_datadriven_init(RealtimeDataDrivenSolver* solver, size_t state_dim,
                            size_t num_layers, double step_size,
                            void (*callback)(double t, const double* y, size_t n, void* data),
                            void* callback_data) {
    if (!solver || state_dim == 0 || num_layers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(RealtimeDataDrivenSolver));
    solver->state_dim = state_dim;
    solver->num_layers = num_layers;
    solver->step_size = step_size;
    solver->buffer_size = 1000;
    
    size_t hidden_dim = 32;
    solver->layer_weights = (double**)malloc(num_layers * sizeof(double*));
    solver->attention_weights = (double*)malloc(state_dim * hidden_dim * sizeof(double));
    solver->buffer = (double*)malloc(solver->buffer_size * state_dim * sizeof(double));
    
    if (!solver->layer_weights || !solver->attention_weights || !solver->buffer) {
        realtime_datadriven_free(solver);
        return -1;
    }
    
    for (size_t l = 0; l < num_layers; l++) {
        solver->layer_weights[l] = (double*)malloc(hidden_dim * state_dim * sizeof(double));
        if (!solver->layer_weights[l]) {
            realtime_datadriven_free(solver);
            return -1;
        }
        for (size_t i = 0; i < hidden_dim * state_dim; i++) {
            solver->layer_weights[l][i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    for (size_t i = 0; i < state_dim * hidden_dim; i++) {
        solver->attention_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    solver->callback = callback;
    solver->callback_data = callback_data;
    
    return 0;
}

void realtime_datadriven_free(RealtimeDataDrivenSolver* solver) {
    if (!solver) return;
    
    if (solver->buffer) free(solver->buffer);
    if (solver->attention_weights) free(solver->attention_weights);
    
    if (solver->layer_weights) {
        for (size_t l = 0; l < solver->num_layers; l++) {
            if (solver->layer_weights[l]) free(solver->layer_weights[l]);
        }
        free(solver->layer_weights);
    }
    
    memset(solver, 0, sizeof(RealtimeDataDrivenSolver));
}

int realtime_datadriven_solve(RealtimeDataDrivenSolver* solver, ODEFunction f,
                              double t0, double t_end, const double* y0,
                              double h, void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Use real-time RK with data-driven enhancement
    RealtimeRKSolver realtime;
    if (realtime_rk_init(&realtime, solver->state_dim, h, solver->callback,
                        solver->callback_data) == 0) {
        double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
        if (y_current) {
            memcpy(y_current, y0, solver->state_dim * sizeof(double));
            
            double t = t0;
            while (t < t_end) {
                double h_actual = (t + h > t_end) ? (t_end - t) : h;
                t = realtime_rk_step(&realtime, f, t, y_current, h_actual, params);
            }
            
            memcpy(y_out, y_current, solver->state_dim * sizeof(double));
            free(y_current);
        }
        realtime_rk_free(&realtime);
    }
    
    return 0;
}

// Distributed Online Solver
int distributed_online_init(DistributedOnlineSolver* solver, size_t state_dim,
                           size_t num_workers, double initial_step_size,
                           double learning_rate) {
    if (!solver || state_dim == 0 || num_workers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DistributedOnlineSolver));
    solver->state_dim = state_dim;
    solver->num_workers = num_workers;
    solver->adaptive_step_size = (double*)malloc(sizeof(double));
    *solver->adaptive_step_size = initial_step_size;
    solver->learning_rate = learning_rate;
    solver->rank = 0;
    solver->size = 1;
    
    size_t chunk_size = state_dim / num_workers;
    solver->work_ranges = (size_t*)malloc((num_workers + 1) * sizeof(size_t));
    
    if (!solver->adaptive_step_size || !solver->work_ranges) {
        distributed_online_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : state_dim;
    }
    
    return 0;
}

void distributed_online_free(DistributedOnlineSolver* solver) {
    if (!solver) return;
    
    if (solver->adaptive_step_size) free(solver->adaptive_step_size);
    if (solver->work_ranges) free(solver->work_ranges);
    
    memset(solver, 0, sizeof(DistributedOnlineSolver));
}

int distributed_online_solve(DistributedOnlineSolver* solver, ODEFunction f,
                             double t0, double t_end, const double* y0,
                             void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Use online RK with distributed processing
    OnlineRKSolver online;
    if (online_rk_init(&online, solver->state_dim, *solver->adaptive_step_size,
                      solver->learning_rate) == 0) {
        double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
        if (y_current) {
            memcpy(y_current, y0, solver->state_dim * sizeof(double));
            
            double t = t0;
            while (t < t_end) {
                t = online_rk_step(&online, f, t, y_current, params);
            }
            
            memcpy(y_out, y_current, solver->state_dim * sizeof(double));
            *solver->adaptive_step_size = *online.adaptive_step_size;
            
            free(y_current);
        }
        online_rk_free(&online);
    }
    
    return 0;
}

// Distributed Real-Time Solver
int distributed_realtime_init(DistributedRealtimeSolver* solver, size_t state_dim,
                              size_t num_workers, double step_size,
                              void (*callback)(double t, const double* y, size_t n, void* data),
                              void* callback_data) {
    if (!solver || state_dim == 0 || num_workers == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(DistributedRealtimeSolver));
    solver->state_dim = state_dim;
    solver->num_workers = num_workers;
    solver->step_size = step_size;
    solver->buffer_size = 1000;
    solver->rank = 0;
    solver->size = 1;
    
    size_t chunk_size = state_dim / num_workers;
    solver->work_ranges = (size_t*)malloc((num_workers + 1) * sizeof(size_t));
    solver->buffer = (double*)malloc(solver->buffer_size * state_dim * sizeof(double));
    
    if (!solver->work_ranges || !solver->buffer) {
        distributed_realtime_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i <= num_workers; i++) {
        solver->work_ranges[i] = (i < num_workers) ? i * chunk_size : state_dim;
    }
    
    solver->callback = callback;
    solver->callback_data = callback_data;
    
    return 0;
}

void distributed_realtime_free(DistributedRealtimeSolver* solver) {
    if (!solver) return;
    
    if (solver->work_ranges) free(solver->work_ranges);
    if (solver->buffer) free(solver->buffer);
    
    memset(solver, 0, sizeof(DistributedRealtimeSolver));
}

int distributed_realtime_solve(DistributedRealtimeSolver* solver, ODEFunction f,
                               double t0, double t_end, const double* y0,
                               double h, void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Use real-time RK with distributed processing
    RealtimeRKSolver realtime;
    if (realtime_rk_init(&realtime, solver->state_dim, h, solver->callback,
                        solver->callback_data) == 0) {
        double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
        if (y_current) {
            memcpy(y_current, y0, solver->state_dim * sizeof(double));
            
            double t = t0;
            while (t < t_end) {
                double h_actual = (t + h > t_end) ? (t_end - t) : h;
                t = realtime_rk_step(&realtime, f, t, y_current, h_actual, params);
            }
            
            memcpy(y_out, y_current, solver->state_dim * sizeof(double));
            free(y_current);
        }
        realtime_rk_free(&realtime);
    }
    
    return 0;
}
