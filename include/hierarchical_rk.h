/*
 * Data-Driven Hierarchical Runge-Kutta Method
 * Transformer-like Architecture for ODE Solving
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef HIERARCHICAL_RK_H
#define HIERARCHICAL_RK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * Hierarchical layer structure for transformer-like ODE solver
 */
typedef struct {
    size_t dimension;           // State dimension
    size_t hidden_dim;          // Hidden dimension for attention
    double* weights;            // Learnable weights
    double* biases;             // Learnable biases
    double* attention_weights; // Attention mechanism weights
} HierarchicalLayer;

/**
 * Hierarchical RK solver structure
 */
typedef struct {
    size_t num_layers;          // Number of hierarchical layers
    size_t state_dim;           // State dimension
    HierarchicalLayer* layers;  // Array of layers
    double learning_rate;       // For adaptive learning
} HierarchicalRKSolver;

/**
 * Initialize hierarchical RK solver
 * 
 * @param solver: Solver structure to initialize
 * @param num_layers: Number of hierarchical layers
 * @param state_dim: Dimension of the state space
 * @param hidden_dim: Hidden dimension for each layer
 * @return: 0 on success, -1 on failure
 */
int hierarchical_rk_init(HierarchicalRKSolver* solver, size_t num_layers, 
                         size_t state_dim, size_t hidden_dim);

/**
 * Free hierarchical RK solver resources
 */
void hierarchical_rk_free(HierarchicalRKSolver* solver);

/**
 * Single step using hierarchical RK method
 * 
 * @param solver: Hierarchical RK solver
 * @param f: ODE function
 * @param t: Current time
 * @param y: Current state (input/output)
 * @param h: Step size
 * @param params: ODE parameters
 * @return: New time value
 */
double hierarchical_rk_step(HierarchicalRKSolver* solver, 
                           void (*f)(double, const double*, double*, void*),
                           double t, double* y, double h, void* params);

/**
 * Solve ODE using hierarchical RK method
 * 
 * @param solver: Hierarchical RK solver
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param h: Step size
 * @param params: ODE parameters
 * @param t_out: Output time array
 * @param y_out: Output state array
 * @return: Number of steps
 */
size_t hierarchical_rk_solve(HierarchicalRKSolver* solver,
                             void (*f)(double, const double*, double*, void*),
                             double t0, double t_end, const double* y0,
                             double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* HIERARCHICAL_RK_H */
