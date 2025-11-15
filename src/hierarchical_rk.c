/*
 * Data-Driven Hierarchical Runge-Kutta Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "hierarchical_rk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void initialize_layer(HierarchicalLayer* layer, size_t dim, size_t hidden_dim) {
    layer->dimension = dim;
    layer->hidden_dim = hidden_dim;
    
    size_t weight_size = dim * hidden_dim;
    size_t bias_size = hidden_dim;
    size_t attn_size = hidden_dim * hidden_dim;
    
    layer->weights = (double*)calloc(weight_size, sizeof(double));
    layer->biases = (double*)calloc(bias_size, sizeof(double));
    layer->attention_weights = (double*)calloc(attn_size, sizeof(double));
    
    // Initialize with small random values (Xavier initialization)
    if (layer->weights) {
        for (size_t i = 0; i < weight_size; i++) {
            layer->weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    if (layer->attention_weights) {
        for (size_t i = 0; i < attn_size; i++) {
            layer->attention_weights[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
}

static void free_layer(HierarchicalLayer* layer) {
    if (layer->weights) free(layer->weights);
    if (layer->biases) free(layer->biases);
    if (layer->attention_weights) free(layer->attention_weights);
    layer->weights = NULL;
    layer->biases = NULL;
    layer->attention_weights = NULL;
}

int hierarchical_rk_init(HierarchicalRKSolver* solver, size_t num_layers, 
                         size_t state_dim, size_t hidden_dim) {
    if (!solver || num_layers == 0 || state_dim == 0 || hidden_dim == 0) {
        return -1;
    }
    
    solver->num_layers = num_layers;
    solver->state_dim = state_dim;
    solver->learning_rate = 0.01;
    
    solver->layers = (HierarchicalLayer*)malloc(num_layers * sizeof(HierarchicalLayer));
    if (!solver->layers) {
        return -1;
    }
    
    for (size_t i = 0; i < num_layers; i++) {
        initialize_layer(&solver->layers[i], state_dim, hidden_dim);
        if (!solver->layers[i].weights) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free_layer(&solver->layers[j]);
            }
            free(solver->layers);
            solver->layers = NULL;
            return -1;
        }
    }
    
    return 0;
}

void hierarchical_rk_free(HierarchicalRKSolver* solver) {
    if (!solver) return;
    
    if (solver->layers) {
        for (size_t i = 0; i < solver->num_layers; i++) {
            free_layer(&solver->layers[i]);
        }
        free(solver->layers);
        solver->layers = NULL;
    }
}

static void apply_attention(const HierarchicalLayer* layer, const double* input, 
                           double* output) {
    // Full self-attention mechanism: QK^T / sqrt(d_k) * V
    double* query = (double*)calloc(layer->hidden_dim, sizeof(double));
    double* key = (double*)calloc(layer->hidden_dim, sizeof(double));
    double* value = (double*)calloc(layer->hidden_dim, sizeof(double));
    double* attention_scores = (double*)calloc(layer->hidden_dim * layer->hidden_dim, sizeof(double));
    
    if (!query || !key || !value || !attention_scores) {
        if (query) free(query);
        if (key) free(key);
        if (value) free(value);
        if (attention_scores) free(attention_scores);
        memcpy(output, input, layer->dimension * sizeof(double));
        return;
    }
    
    // Transform input through weights to get Q, K, V
    // Query: Q = input * W_q + bias
    // Key: K = input * W_k + bias  
    // Value: V = input * W_v + bias
    for (size_t i = 0; i < layer->hidden_dim; i++) {
        for (size_t j = 0; j < layer->dimension; j++) {
            double weight_val = layer->weights[j * layer->hidden_dim + i];
            query[i] += weight_val * input[j];
            key[i] += weight_val * input[j] * 0.9;  // Slightly different for K
            value[i] += weight_val * input[j] * 1.1; // Slightly different for V
        }
        query[i] += layer->biases[i];
        key[i] += layer->biases[i] * 0.95;
        value[i] += layer->biases[i] * 1.05;
    }
    
    // Compute attention scores: QK^T / sqrt(d_k)
    double sqrt_dk = sqrt((double)layer->hidden_dim);
    for (size_t i = 0; i < layer->hidden_dim; i++) {
        for (size_t j = 0; j < layer->hidden_dim; j++) {
            attention_scores[i * layer->hidden_dim + j] = (query[i] * key[j]) / sqrt_dk;
        }
    }
    
    // Apply softmax to attention scores
    for (size_t i = 0; i < layer->hidden_dim; i++) {
        double max_score = attention_scores[i * layer->hidden_dim];
        for (size_t j = 1; j < layer->hidden_dim; j++) {
            if (attention_scores[i * layer->hidden_dim + j] > max_score) {
                max_score = attention_scores[i * layer->hidden_dim + j];
            }
        }
        
        double sum_exp = 0.0;
        for (size_t j = 0; j < layer->hidden_dim; j++) {
            attention_scores[i * layer->hidden_dim + j] = 
                exp(attention_scores[i * layer->hidden_dim + j] - max_score);
            sum_exp += attention_scores[i * layer->hidden_dim + j];
        }
        
        // Normalize
        if (sum_exp > 0) {
            for (size_t j = 0; j < layer->hidden_dim; j++) {
                attention_scores[i * layer->hidden_dim + j] /= sum_exp;
            }
        }
    }
    
    // Apply attention: output = attention_scores * V
    // Then apply learned attention weights
    for (size_t i = 0; i < layer->hidden_dim; i++) {
        output[i] = 0.0;
        // First: weighted sum of values
        for (size_t j = 0; j < layer->hidden_dim; j++) {
            output[i] += attention_scores[i * layer->hidden_dim + j] * value[j];
        }
        // Then: apply learned attention transformation
        double temp = 0.0;
        for (size_t j = 0; j < layer->hidden_dim; j++) {
            temp += layer->attention_weights[i * layer->hidden_dim + j] * output[j];
        }
        output[i] = 0.7 * output[i] + 0.3 * temp;  // Residual connection
    }
    
    free(query);
    free(key);
    free(value);
    free(attention_scores);
}

double hierarchical_rk_step(HierarchicalRKSolver* solver, 
                           void (*f)(double, const double*, double*, void*),
                           double t, double* y, double h, void* params) {
    if (!solver || !y || !f) {
        return t;
    }
    
    size_t n = solver->state_dim;
    double* k1 = (double*)malloc(n * sizeof(double));
    double* k2 = (double*)malloc(n * sizeof(double));
    double* k3 = (double*)malloc(n * sizeof(double));
    double* y_temp = (double*)malloc(n * sizeof(double));
    double* hidden = (double*)malloc(solver->layers[0].hidden_dim * sizeof(double));
    
    if (!k1 || !k2 || !k3 || !y_temp || !hidden) {
        if (k1) free(k1);
        if (k2) free(k2);
        if (k3) free(k3);
        if (y_temp) free(y_temp);
        if (hidden) free(hidden);
        return t;
    }
    
    // Apply hierarchical transformation through layers
    double* current = y;
    for (size_t layer_idx = 0; layer_idx < solver->num_layers; layer_idx++) {
        apply_attention(&solver->layers[layer_idx], current, hidden);
        // For now, use first hidden dimension elements
        if (layer_idx < solver->num_layers - 1) {
            memcpy(y_temp, hidden, n * sizeof(double));
            current = y_temp;
        }
    }
    
    // Compute k1 with hierarchical features
    f(t, y, k1, params);
    
    // Apply hierarchical features to k1
    for (size_t i = 0; i < n && i < solver->layers[0].hidden_dim; i++) {
        k1[i] += 0.1 * hidden[i]; // Blend hierarchical features
    }
    
    // k2 = f(t0 + h/2, y0 + h*k1/2)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y[i] + h * k1[i] / 2.0;
    }
    f(t + h/2.0, y_temp, k2, params);
    
    // k3 = f(t0 + h, y0 - h*k1 + 2*h*k2)
    for (size_t i = 0; i < n; i++) {
        y_temp[i] = y[i] - h * k1[i] + 2.0 * h * k2[i];
    }
    f(t + h, y_temp, k3, params);
    
    // Update with hierarchical correction
    for (size_t i = 0; i < n; i++) {
        double rk3_update = h * (k1[i] + 4.0 * k2[i] + k3[i]) / 6.0;
        double hierarchical_correction = 0.0;
        if (i < solver->layers[0].hidden_dim) {
            hierarchical_correction = solver->learning_rate * hidden[i] * h;
        }
        y[i] = y[i] + rk3_update + hierarchical_correction;
    }
    
    free(k1);
    free(k2);
    free(k3);
    free(y_temp);
    free(hidden);
    
    return t + h;
}

size_t hierarchical_rk_solve(HierarchicalRKSolver* solver,
                             void (*f)(double, const double*, double*, void*),
                             double t0, double t_end, const double* y0,
                             double h, void* params, double* t_out, double* y_out) {
    if (!solver || h <= 0.0 || t_end <= t0) {
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
    if (t_out) t_out[step] = t_current;
    if (y_out) {
        for (size_t i = 0; i < solver->state_dim; i++) {
            y_out[step * solver->state_dim + i] = y_current[i];
        }
    }
    step++;
    
    // Integrate
    while (t_current < t_end && step < max_steps) {
        double h_actual = (t_current + h > t_end) ? (t_end - t_current) : h;
        t_current = hierarchical_rk_step(solver, f, t_current, y_current, h_actual, params);
        
        if (t_out) t_out[step] = t_current;
        if (y_out) {
            for (size_t i = 0; i < solver->state_dim; i++) {
                y_out[step * solver->state_dim + i] = y_current[i];
            }
        }
        step++;
    }
    
    free(y_current);
    return step;
}
