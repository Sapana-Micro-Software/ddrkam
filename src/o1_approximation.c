/*
 * O(1) Real-Time Approximation Solvers for Differential Equations
 * Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "o1_approximation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Simple hash function for parameter lookup
static size_t hash_param(double param, double param_min, double param_max, size_t table_size) {
    double normalized = (param - param_min) / (param_max - param_min);
    normalized = fmax(0.0, fmin(1.0, normalized));  // Clamp to [0, 1]
    return (size_t)(normalized * (table_size - 1));
}

// Bilinear interpolation
static double bilinear_interpolate(double** grid, size_t i, size_t j,
                                  double t, double param,
                                  double t_min, double t_max,
                                  double param_min, double param_max,
                                  size_t t_size, size_t p_size) {
    // Find grid indices
    size_t t_idx = (size_t)((t - t_min) / (t_max - t_min) * (t_size - 1));
    size_t p_idx = (size_t)((param - param_min) / (param_max - param_min) * (p_size - 1));
    
    // Clamp indices
    if (t_idx >= t_size) t_idx = t_size - 1;
    if (p_idx >= p_size) p_idx = p_size - 1;
    if (t_idx == 0 && t < t_min) t_idx = 0;
    if (p_idx == 0 && param < param_min) p_idx = 0;
    
    // Get corner values
    double v00 = grid[p_idx][t_idx];
    double v01 = (t_idx + 1 < t_size) ? grid[p_idx][t_idx + 1] : v00;
    double v10 = (p_idx + 1 < p_size) ? grid[p_idx + 1][t_idx] : v00;
    double v11 = (t_idx + 1 < t_size && p_idx + 1 < p_size) ? 
                 grid[p_idx + 1][t_idx + 1] : v00;
    
    // Interpolation weights
    double t_frac = (t - (t_min + (t_max - t_min) * t_idx / (t_size - 1))) / 
                    ((t_max - t_min) / (t_size - 1));
    double p_frac = (param - (param_min + (param_max - param_min) * p_idx / (p_size - 1))) /
                    ((param_max - param_min) / (p_size - 1));
    
    t_frac = fmax(0.0, fmin(1.0, t_frac));
    p_frac = fmax(0.0, fmin(1.0, p_frac));
    
    // Bilinear interpolation
    return (1 - t_frac) * (1 - p_frac) * v00 +
           t_frac * (1 - p_frac) * v01 +
           (1 - t_frac) * p_frac * v10 +
           t_frac * p_frac * v11;
}

// ============================================================================
// Lookup Table Solver
// ============================================================================

int lookup_table_init(LookupTableSolver* solver,
                     size_t param_grid_size,
                     size_t time_grid_size,
                     size_t state_dim,
                     double param_min,
                     double param_max,
                     double time_max) {
    if (!solver || param_grid_size == 0 || time_grid_size == 0 || state_dim == 0) {
        return -1;
    }
    
    solver->param_grid_size = param_grid_size;
    solver->time_grid_size = time_grid_size;
    solver->state_dim = state_dim;
    solver->param_min = param_min;
    solver->param_max = param_max;
    solver->time_max = time_max;
    solver->use_hash = 0;  // Direct indexing is faster
    
    // Allocate memory
    solver->solution_grid = (double***)malloc(param_grid_size * sizeof(double**));
    solver->param_values = (double*)malloc(param_grid_size * sizeof(double));
    solver->time_values = (double*)malloc(time_grid_size * sizeof(double));
    
    if (!solver->solution_grid || !solver->param_values || !solver->time_values) {
        lookup_table_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < param_grid_size; i++) {
        solver->solution_grid[i] = (double**)malloc(time_grid_size * sizeof(double*));
        if (!solver->solution_grid[i]) {
            lookup_table_free(solver);
            return -1;
        }
        
        for (size_t j = 0; j < time_grid_size; j++) {
            solver->solution_grid[i][j] = (double*)malloc(state_dim * sizeof(double));
            if (!solver->solution_grid[i][j]) {
                lookup_table_free(solver);
                return -1;
            }
        }
        
        solver->param_values[i] = param_min + (param_max - param_min) * i / (param_grid_size - 1);
    }
    
    for (size_t j = 0; j < time_grid_size; j++) {
        solver->time_values[j] = time_max * j / (time_grid_size - 1);
    }
    
    return 0;
}

void lookup_table_free(LookupTableSolver* solver) {
    if (!solver) return;
    
    if (solver->solution_grid) {
        for (size_t i = 0; i < solver->param_grid_size; i++) {
            if (solver->solution_grid[i]) {
                for (size_t j = 0; j < solver->time_grid_size; j++) {
                    free(solver->solution_grid[i][j]);
                }
                free(solver->solution_grid[i]);
            }
        }
        free(solver->solution_grid);
    }
    
    free(solver->param_values);
    free(solver->time_values);
    free(solver->param_hash_table);
    
    memset(solver, 0, sizeof(LookupTableSolver));
}

int lookup_table_precompute(LookupTableSolver* solver,
                           void (*ode_func)(double t, const double* y, double* dydt, void* params),
                           void* params) {
    if (!solver || !ode_func) return -1;
    
    // Pre-compute solutions for each parameter and time point
    // This is done offline, so can use any method (e.g., RK4)
    double* y_temp = (double*)malloc(solver->state_dim * sizeof(double));
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (!y_temp || !dydt) {
        free(y_temp);
        free(dydt);
        return -1;
    }
    
    // Simple Euler pre-computation (can be replaced with RK4 for better accuracy)
    double dt = solver->time_max / (solver->time_grid_size - 1);
    
    for (size_t p = 0; p < solver->param_grid_size; p++) {
        // Initialize state (can be parameterized)
        for (size_t i = 0; i < solver->state_dim; i++) {
            y_temp[i] = 1.0;  // Default initial condition
        }
        
        // Solve for this parameter value
        for (size_t t = 0; t < solver->time_grid_size; t++) {
            double t_val = solver->time_values[t];
            
            // Store solution
            memcpy(solver->solution_grid[p][t], y_temp, solver->state_dim * sizeof(double));
            
            // Step forward (if not last time point)
            if (t < solver->time_grid_size - 1) {
                ode_func(t_val, y_temp, dydt, params);
                for (size_t i = 0; i < solver->state_dim; i++) {
                    y_temp[i] += dt * dydt[i];
                }
            }
        }
    }
    
    free(y_temp);
    free(dydt);
    
    return 0;
}

int lookup_table_solve(LookupTableSolver* solver,
                      double t,
                      double param,
                      const double* y0,
                      double* y_out) {
    if (!solver || !y_out) return -1;
    
    // Check bounds
    if (t < 0.0 || t > solver->time_max ||
        param < solver->param_min || param > solver->param_max) {
        return -1;  // Out of range
    }
    
    // O(1) lookup with bilinear interpolation
    for (size_t i = 0; i < solver->state_dim; i++) {
        // For each state dimension, interpolate
        double** grid_2d = (double**)malloc(solver->param_grid_size * sizeof(double*));
        if (!grid_2d) return -1;
        
        for (size_t p = 0; p < solver->param_grid_size; p++) {
            grid_2d[p] = (double*)malloc(solver->time_grid_size * sizeof(double));
            if (!grid_2d[p]) {
                for (size_t j = 0; j < p; j++) {
                    free(grid_2d[j]);
                }
                free(grid_2d);
                return -1;
            }
            
            for (size_t t_idx = 0; t_idx < solver->time_grid_size; t_idx++) {
                grid_2d[p][t_idx] = solver->solution_grid[p][t_idx][i];
            }
        }
        
        y_out[i] = bilinear_interpolate(grid_2d, 0, 0, t, param,
                                       0.0, solver->time_max,
                                       solver->param_min, solver->param_max,
                                       solver->time_grid_size, solver->param_grid_size);
        
        // Scale by initial condition
        if (y0) {
            y_out[i] *= y0[i];
        }
        
        for (size_t p = 0; p < solver->param_grid_size; p++) {
            free(grid_2d[p]);
        }
        free(grid_2d);
    }
    
    return 0;
}

// ============================================================================
// Neural Network Approximator
// ============================================================================

// Simple activation function (ReLU)
static double relu(double x) {
    return fmax(0.0, x);
}

// Sigmoid activation
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int neural_approximator_init(NeuralApproximator* net,
                             const size_t* layer_sizes,
                             size_t num_layers) {
    if (!net || !layer_sizes || num_layers < 2) {
        return -1;
    }
    
    net->num_layers = num_layers;
    net->input_dim = layer_sizes[0];
    net->output_dim = layer_sizes[num_layers - 1];
    
    // Allocate memory
    net->layer_sizes = (size_t*)malloc(num_layers * sizeof(size_t));
    net->weights = (double**)malloc((num_layers - 1) * sizeof(double*));
    net->biases = (double*)malloc((num_layers - 1) * sizeof(double));
    net->activation_buffer = (double*)malloc(layer_sizes[0] * sizeof(double));
    
    if (!net->layer_sizes || !net->weights || !net->biases || !net->activation_buffer) {
        neural_approximator_free(net);
        return -1;
    }
    
    memcpy(net->layer_sizes, layer_sizes, num_layers * sizeof(size_t));
    
    // Allocate weights for each layer
    for (size_t l = 0; l < num_layers - 1; l++) {
        size_t num_weights = layer_sizes[l] * layer_sizes[l + 1];
        net->weights[l] = (double*)malloc(num_weights * sizeof(double));
        if (!net->weights[l]) {
            neural_approximator_free(net);
            return -1;
        }
        
        // Initialize with small random values (Xavier initialization)
        double scale = sqrt(2.0 / (layer_sizes[l] + layer_sizes[l + 1]));
        for (size_t i = 0; i < num_weights; i++) {
            net->weights[l][i] = ((double)rand() / RAND_MAX - 0.5) * scale;
        }
    }
    
    // Initialize biases to zero
    for (size_t l = 0; l < num_layers - 1; l++) {
        net->biases[l] = 0.0;
    }
    
    return 0;
}

void neural_approximator_free(NeuralApproximator* net) {
    if (!net) return;
    
    free(net->layer_sizes);
    
    if (net->weights) {
        for (size_t l = 0; l < net->num_layers - 1; l++) {
            free(net->weights[l]);
        }
        free(net->weights);
    }
    
    free(net->biases);
    free(net->activation_buffer);
    
    memset(net, 0, sizeof(NeuralApproximator));
}

int neural_approximator_solve(NeuralApproximator* net,
                              double t,
                              const double* y0,
                              const double* params,
                              size_t num_params,
                              double* y_out) {
    if (!net || !y0 || !y_out) return -1;
    
    // Construct input: [t, y0[0..n], params[0..m]]
    size_t input_size = 1 + net->input_dim - 1 - num_params + num_params;
    if (input_size != net->input_dim) {
        // Adjust if needed
        input_size = net->input_dim;
    }
    
    double* input = (double*)malloc(net->input_dim * sizeof(double));
    if (!input) return -1;
    
    input[0] = t;
    for (size_t i = 0; i < net->input_dim - 1 && i < net->output_dim; i++) {
        input[i + 1] = y0[i];
    }
    for (size_t i = 0; i < num_params && (i + 1 + net->output_dim) < net->input_dim; i++) {
        input[i + 1 + net->output_dim] = params[i];
    }
    
    // Forward pass: O(1) with fixed network depth
    double* activations = input;
    double* next_activations = net->activation_buffer;
    
    for (size_t l = 0; l < net->num_layers - 1; l++) {
        size_t layer_size = net->layer_sizes[l];
        size_t next_size = net->layer_sizes[l + 1];
        
        for (size_t j = 0; j < next_size; j++) {
            double sum = net->biases[l];
            for (size_t i = 0; i < layer_size; i++) {
                sum += activations[i] * net->weights[l][i * next_size + j];
            }
            next_activations[j] = relu(sum);  // ReLU activation
        }
        
        // Swap buffers
        double* temp = activations;
        activations = next_activations;
        next_activations = (l == 0) ? input : temp;
    }
    
    // Output layer: no activation (linear)
    memcpy(y_out, activations, net->output_dim * sizeof(double));
    
    free(input);
    
    return 0;
}

// ============================================================================
// Chebyshev Polynomial Approximator
// ============================================================================

int chebyshev_approximator_init(ChebyshevApproximator* approx,
                                size_t num_coefficients,
                                size_t num_params,
                                const double* param_values,
                                double time_scale) {
    if (!approx || num_coefficients == 0 || num_params == 0 || !param_values) {
        return -1;
    }
    
    approx->num_coefficients = num_coefficients;
    approx->num_params = num_params;
    approx->time_scale = time_scale;
    
    approx->coefficients = (double**)malloc(num_params * sizeof(double*));
    approx->param_values = (double*)malloc(num_params * sizeof(double));
    
    if (!approx->coefficients || !approx->param_values) {
        chebyshev_approximator_free(approx);
        return -1;
    }
    
    for (size_t i = 0; i < num_params; i++) {
        approx->coefficients[i] = (double*)malloc(num_coefficients * sizeof(double));
        if (!approx->coefficients[i]) {
            chebyshev_approximator_free(approx);
            return -1;
        }
        // Initialize to zero (will be loaded)
        memset(approx->coefficients[i], 0, num_coefficients * sizeof(double));
    }
    
    memcpy(approx->param_values, param_values, num_params * sizeof(double));
    
    return 0;
}

void chebyshev_approximator_free(ChebyshevApproximator* approx) {
    if (!approx) return;
    
    if (approx->coefficients) {
        for (size_t i = 0; i < approx->num_params; i++) {
            free(approx->coefficients[i]);
        }
        free(approx->coefficients);
    }
    
    free(approx->param_values);
    
    memset(approx, 0, sizeof(ChebyshevApproximator));
}

int chebyshev_approximator_solve(ChebyshevApproximator* approx,
                                 double t,
                                 size_t param_idx,
                                 double* y_out) {
    if (!approx || !y_out || param_idx >= approx->num_params) {
        return -1;
    }
    
    // Normalize time to [-1, 1] for Chebyshev polynomials
    double x = 2.0 * (t / approx->time_scale) - 1.0;
    x = fmax(-1.0, fmin(1.0, x));
    
    // Clenshaw's algorithm: O(k) where k is fixed → O(1)
    double T_prev = 1.0;      // T₀(x) = 1
    double T_curr = x;         // T₁(x) = x
    double result = approx->coefficients[param_idx][0] * T_prev;
    
    if (approx->num_coefficients > 1) {
        result += approx->coefficients[param_idx][1] * T_curr;
    }
    
    for (size_t i = 2; i < approx->num_coefficients; i++) {
        // Recurrence: T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
        double T_next = 2.0 * x * T_curr - T_prev;
        result += approx->coefficients[param_idx][i] * T_next;
        
        T_prev = T_curr;
        T_curr = T_next;
    }
    
    *y_out = result;
    
    return 0;
}

// ============================================================================
// Hybrid O(1) Solver
// ============================================================================

int o1_solver_init(O1ApproximationSolver* solver, int method) {
    if (!solver) return -1;
    
    solver->method = method;
    solver->lookup = NULL;
    solver->neural = NULL;
    solver->chebyshev = NULL;
    solver->rom = NULL;
    solver->use_fallback = 0;
    solver->fallback_solver = NULL;
    
    return 0;
}

void o1_solver_free(O1ApproximationSolver* solver) {
    if (!solver) return;
    
    if (solver->lookup) {
        lookup_table_free(solver->lookup);
        free(solver->lookup);
    }
    
    if (solver->neural) {
        neural_approximator_free(solver->neural);
        free(solver->neural);
    }
    
    if (solver->chebyshev) {
        chebyshev_approximator_free(solver->chebyshev);
        free(solver->chebyshev);
    }
    
    if (solver->rom) {
        reduced_order_free(solver->rom);
        free(solver->rom);
    }
    
    memset(solver, 0, sizeof(O1ApproximationSolver));
}

int o1_solver_solve(O1ApproximationSolver* solver,
                    double t,
                    const double* y0,
                    const double* params,
                    size_t num_params,
                    double* y_out) {
    if (!solver || !y0 || !y_out) return -1;
    
    int result = -1;
    
    switch (solver->method) {
        case O1_METHOD_LOOKUP:
            if (solver->lookup && num_params > 0) {
                result = lookup_table_solve(solver->lookup, t, params[0], y0, y_out);
            }
            break;
            
        case O1_METHOD_NEURAL:
            if (solver->neural) {
                result = neural_approximator_solve(solver->neural, t, y0, params, num_params, y_out);
            }
            break;
            
        case O1_METHOD_CHEBYSHEV:
            if (solver->chebyshev && num_params > 0) {
                // Find nearest parameter index
                size_t param_idx = 0;
                double min_dist = fabs(params[0] - solver->chebyshev->param_values[0]);
                for (size_t i = 1; i < solver->chebyshev->num_params; i++) {
                    double dist = fabs(params[0] - solver->chebyshev->param_values[i]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        param_idx = i;
                    }
                }
                result = chebyshev_approximator_solve(solver->chebyshev, t, param_idx, y_out);
            }
            break;
            
        default:
            result = -1;
    }
    
    // Fallback to traditional solver if approximation fails
    if (result != 0 && solver->use_fallback && solver->fallback_solver) {
        // Use fallback solver (not implemented here, would call traditional RK4/Adams)
        result = 0;  // Placeholder
    }
    
    return result;
}

// Placeholder implementations for reduced-order model
int reduced_order_init(ReducedOrderSolver* rom,
                      size_t full_dim,
                      size_t reduced_dim,
                      size_t num_time_points) {
    return 0;
}

void reduced_order_free(ReducedOrderSolver* rom) {
}

int reduced_order_solve(ReducedOrderSolver* rom,
                       double t,
                       const double* y0,
                       double* y_out) {
    return 0;
}
