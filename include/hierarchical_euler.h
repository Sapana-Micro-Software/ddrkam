/*
 * Data-Driven Euler's Method (Hierarchical/Transformer-Inspired)
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef HIERARCHICAL_EULER_H
#define HIERARCHICAL_EULER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * Function pointer type for the differential equation system
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Data-Driven Euler Solver Structure
 */
typedef struct {
    size_t state_dim;        // Dimension of state vector
    size_t num_layers;       // Number of hierarchical layers
    size_t hidden_dim;       // Hidden dimension for each layer
    
    // Hierarchical layers (transformer-inspired)
    double** layer_weights;  // [num_layers][hidden_dim * state_dim]
    double** layer_biases;   // [num_layers][hidden_dim]
    double* attention_weights; // Attention mechanism weights
    
    // Learning parameters
    double learning_rate;   // Learning rate for adaptive refinement
    
    // Internal state
    double* hidden_state;    // Current hidden state
} HierarchicalEulerSolver;

/**
 * Initialize Data-Driven Euler solver
 * 
 * @param solver: Solver structure to initialize
 * @param num_layers: Number of hierarchical layers (typically 2-4)
 * @param state_dim: Dimension of the ODE system
 * @param hidden_dim: Hidden dimension for hierarchical processing (typically 16-64)
 * @return: 0 on success, -1 on error
 */
int hierarchical_euler_init(HierarchicalEulerSolver* solver, size_t num_layers, 
                            size_t state_dim, size_t hidden_dim);

/**
 * Free Data-Driven Euler solver resources
 */
void hierarchical_euler_free(HierarchicalEulerSolver* solver);

/**
 * Data-Driven Euler step
 * Combines standard Euler with hierarchical refinement
 * 
 * @param solver: Initialized solver
 * @param f: ODE function
 * @param t: Current time
 * @param y: Current state (input/output)
 * @param h: Step size
 * @param params: User parameters
 * @return: New time (t + h)
 */
double hierarchical_euler_step(HierarchicalEulerSolver* solver, ODEFunction f,
                               double t, double* y, double h, void* params);

/**
 * Solve ODE using Data-Driven Euler method
 * 
 * @param solver: Initialized solver
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param h: Step size
 * @param params: User parameters
 * @param t_out: Output time array
 * @param y_out: Output state array
 * @return: Number of steps
 */
size_t hierarchical_euler_solve(HierarchicalEulerSolver* solver, ODEFunction f,
                                double t0, double t_end, const double* y0,
                                double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* HIERARCHICAL_EULER_H */
