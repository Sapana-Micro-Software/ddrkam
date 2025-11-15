/*
 * Data-Driven Euler's Method Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "hierarchical_euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int hierarchical_euler_init(HierarchicalEulerSolver* solver, size_t num_layers,
                            size_t state_dim, size_t hidden_dim) {
    if (!solver || num_layers == 0 || state_dim == 0 || hidden_dim == 0) {
        return -1;
    }
    
    solver->state_dim = state_dim;
    solver->num_layers = num_layers;
    solver->hidden_dim = hidden_dim;
    solver->learning_rate = 0.01;
    
    // Allocate layer weights
    solver->layer_weights = (double**)malloc(num_layers * sizeof(double*));
    solver->layer_biases = (double**)malloc(num_layers * sizeof(double*));
    
    if (!solver->layer_weights || !solver->layer_biases) {
        hierarchical_euler_free(solver);
        return -1;
    }
    
    // Initialize layers
    for (size_t l = 0; l < num_layers; l++) {
        size_t weight_size = hidden_dim * state_dim;
        solver->layer_weights[l] = (double*)malloc(weight_size * sizeof(double));
        solver->layer_biases[l] = (double*)malloc(hidden_dim * sizeof(double));
        
        if (!solver->layer_weights[l] || !solver->layer_biases[l]) {
            hierarchical_euler_free(solver);
            return -1;
        }
        
        // Initialize weights with small random values
        for (size_t i = 0; i < weight_size; i++) {
            solver->layer_weights[l][i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
        for (size_t i = 0; i < hidden_dim; i++) {
            solver->layer_biases[l][i] = ((double)rand() / RAND_MAX - 0.5) * 0.05;
        }
    }
    
    // Allocate attention weights
    size_t attention_size = state_dim * hidden_dim;
    solver->attention_weights = (double*)malloc(attention_size * sizeof(double));
    if (!solver->attention_weights) {
        hierarchical_euler_free(solver);
        return -1;
    }
    
    // Initialize attention weights
    for (size_t i = 0; i < attention_size; i++) {
        solver->attention_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    // Allocate hidden state
    solver->hidden_state = (double*)malloc(hidden_dim * sizeof(double));
    if (!solver->hidden_state) {
        hierarchical_euler_free(solver);
        return -1;
    }
    
    memset(solver->hidden_state, 0, hidden_dim * sizeof(double));
    
    return 0;
}

void hierarchical_euler_free(HierarchicalEulerSolver* solver) {
    if (!solver) return;
    
    if (solver->layer_weights) {
        for (size_t l = 0; l < solver->num_layers; l++) {
            if (solver->layer_weights[l]) {
                free(solver->layer_weights[l]);
            }
        }
        free(solver->layer_weights);
    }
    
    if (solver->layer_biases) {
        for (size_t l = 0; l < solver->num_layers; l++) {
            if (solver->layer_biases[l]) {
                free(solver->layer_biases[l]);
            }
        }
        free(solver->layer_biases);
    }
    
    if (solver->attention_weights) {
        free(solver->attention_weights);
    }
    
    if (solver->hidden_state) {
        free(solver->hidden_state);
    }
    
    memset(solver, 0, sizeof(HierarchicalEulerSolver));
}

double hierarchical_euler_step(HierarchicalEulerSolver* solver, ODEFunction f,
                               double t, double* y, double h, void* params) {
    if (!solver || !f || !y) {
        return t;
    }
    
    // Standard Euler step: compute derivative
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    if (!dydt) {
        return t;
    }
    
    f(t, y, dydt, params);
    
    // Standard Euler update: y_euler = y + h * dydt
    double* y_euler = (double*)malloc(solver->state_dim * sizeof(double));
    if (!y_euler) {
        free(dydt);
        return t;
    }
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        y_euler[i] = y[i] + h * dydt[i];
    }
    
    // Hierarchical refinement: process through transformer-like layers
    double* layer_input = (double*)malloc(solver->hidden_dim * sizeof(double));
    if (!layer_input) {
        free(dydt);
        free(y_euler);
        return t;
    }
    
    // Project state to hidden dimension
    for (size_t i = 0; i < solver->hidden_dim; i++) {
        layer_input[i] = 0.0;
        for (size_t j = 0; j < solver->state_dim; j++) {
            size_t idx = i * solver->state_dim + j;
            layer_input[i] += solver->attention_weights[idx] * y[j];
        }
    }
    
    // Process through hierarchical layers
    double* current = layer_input;
    for (size_t l = 0; l < solver->num_layers; l++) {
        double* next = (double*)malloc(solver->hidden_dim * sizeof(double));
        if (!next) {
            free(dydt);
            free(y_euler);
            free(layer_input);
            return t;
        }
        
        // Linear transformation: W * x + b
        for (size_t i = 0; i < solver->hidden_dim; i++) {
            next[i] = solver->layer_biases[l][i];
            for (size_t j = 0; j < solver->hidden_dim; j++) {
                size_t idx = i * solver->hidden_dim + j;
                next[i] += solver->layer_weights[l][idx] * current[j];
            }
        }
        
        // ReLU activation
        for (size_t i = 0; i < solver->hidden_dim; i++) {
            next[i] = (next[i] > 0) ? next[i] : 0.0;
        }
        
        if (current != layer_input) {
            free(current);
        }
        current = next;
    }
    
    // Project back to state dimension and apply correction
    double* correction = (double*)malloc(solver->state_dim * sizeof(double));
    if (!correction) {
        free(dydt);
        free(y_euler);
        free(layer_input);
        if (current != layer_input) free(current);
        return t;
    }
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        correction[i] = 0.0;
        for (size_t j = 0; j < solver->hidden_dim; j++) {
            size_t idx = i * solver->hidden_dim + j;
            correction[i] += solver->attention_weights[idx] * current[j];
        }
        correction[i] *= solver->learning_rate * h;
    }
    
    // Combine Euler step with hierarchical correction
    // y_final = y_euler + correction
    for (size_t i = 0; i < solver->state_dim; i++) {
        y[i] = y_euler[i] + correction[i];
    }
    
    // Update hidden state
    memcpy(solver->hidden_state, current, solver->hidden_dim * sizeof(double));
    
    // Cleanup
    free(dydt);
    free(y_euler);
    free(layer_input);
    if (current != layer_input) {
        free(current);
    }
    free(correction);
    
    return t + h;
}

size_t hierarchical_euler_solve(HierarchicalEulerSolver* solver, ODEFunction f,
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
    
    // Integrate using Data-Driven Euler
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = hierarchical_euler_step(solver, f, t_current, y_current, h_actual, params);
        
        t_out[step] = t_current;
        for (size_t i = 0; i < solver->state_dim; i++) {
            y_out[step * solver->state_dim + i] = y_current[i];
        }
        step++;
    }
    
    free(y_current);
    return step;
}
