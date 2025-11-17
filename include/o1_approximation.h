/*
 * O(1) Real-Time Approximation Solvers for Differential Equations
 * Constant-time approximation methods for hard real-time applications
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef O1_APPROXIMATION_H
#define O1_APPROXIMATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * Lookup table-based O(1) solver
 * Pre-computed solutions with bilinear interpolation
 */
typedef struct {
    double** solution_grid;      // [param_idx][time_idx][state_dim]
    double* param_values;        // Parameter grid points
    double* time_values;         // Time grid points
    size_t param_grid_size;      // Number of parameter grid points
    size_t time_grid_size;       // Number of time grid points
    size_t state_dim;            // System dimension
    double param_min, param_max; // Parameter range
    double time_max;             // Maximum time
    int use_hash;                // Use hash table for parameter lookup
    size_t* param_hash_table;    // Hash table for O(1) param lookup
} LookupTableSolver;

/**
 * Neural network approximator for O(1) solutions
 */
typedef struct {
    double** weights;            // Network weights [layer][neuron]
    double* biases;              // Network biases
    size_t* layer_sizes;        // [input_dim, hidden1, ..., output_dim]
    size_t num_layers;          // Number of layers
    size_t input_dim;           // Input: [t, y0[0..n], params[0..m]]
    size_t output_dim;          // Output: y(t)
    double* activation_buffer;  // Buffer for activations
} NeuralApproximator;

/**
 * Chebyshev polynomial approximator
 */
typedef struct {
    double** coefficients;      // [param_idx][coeff_idx]
    size_t num_coefficients;    // Polynomial degree + 1
    size_t num_params;          // Number of parameter sets
    double* param_values;       // Parameter values
    double time_scale;           // Time scaling factor
} ChebyshevApproximator;

/**
 * Reduced-order model (ROM) solver
 */
typedef struct {
    double** projection_matrix;  // [n × r] projection matrix
    double** reduced_solution;   // Pre-computed reduced solutions
    size_t full_dim;            // Full system dimension
    size_t reduced_dim;         // Reduced dimension (r << n)
    size_t num_time_points;      // Number of pre-computed time points
    double* time_points;        // Time points
} ReducedOrderSolver;

/**
 * Hybrid O(1) solver combining multiple methods
 */
typedef struct {
    enum {
        O1_METHOD_LOOKUP,
        O1_METHOD_NEURAL,
        O1_METHOD_CHEBYSHEV,
        O1_METHOD_REDUCED_ORDER,
        O1_METHOD_AUTO  // Automatically select best method
    } method;
    
    LookupTableSolver* lookup;
    NeuralApproximator* neural;
    ChebyshevApproximator* chebyshev;
    ReducedOrderSolver* rom;
    
    // Fallback to traditional solver if approximation fails
    int use_fallback;
    void* fallback_solver;
} O1ApproximationSolver;

// ============================================================================
// Lookup Table Solver
// ============================================================================

/**
 * Initialize lookup table solver
 * 
 * @param solver: Solver structure
 * @param param_grid_size: Number of parameter grid points
 * @param time_grid_size: Number of time grid points
 * @param state_dim: System dimension
 * @param param_min, param_max: Parameter range
 * @param time_max: Maximum time
 * @return: 0 on success, -1 on failure
 */
int lookup_table_init(LookupTableSolver* solver,
                     size_t param_grid_size,
                     size_t time_grid_size,
                     size_t state_dim,
                     double param_min,
                     double param_max,
                     double time_max);

/**
 * Pre-compute solution grid (offline)
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function to solve
 * @param params: ODE parameters
 * @return: 0 on success, -1 on failure
 */
int lookup_table_precompute(LookupTableSolver* solver,
                           void (*ode_func)(double t, const double* y, double* dydt, void* params),
                           void* params);

/**
 * O(1) solution lookup with bilinear interpolation
 * 
 * @param solver: Solver structure
 * @param t: Time point
 * @param param: Parameter value
 * @param y0: Initial condition
 * @param y_out: Output solution (must be allocated, size = state_dim)
 * @return: 0 on success, -1 if out of range
 */
int lookup_table_solve(LookupTableSolver* solver,
                      double t,
                      double param,
                      const double* y0,
                      double* y_out);

/**
 * Free lookup table solver
 */
void lookup_table_free(LookupTableSolver* solver);

// ============================================================================
// Neural Network Approximator
// ============================================================================

/**
 * Initialize neural network approximator
 * 
 * @param net: Network structure
 * @param layer_sizes: Array of layer sizes [input, hidden1, ..., output]
 * @param num_layers: Number of layers
 * @return: 0 on success, -1 on failure
 */
int neural_approximator_init(NeuralApproximator* net,
                             const size_t* layer_sizes,
                             size_t num_layers);

/**
 * Load pre-trained weights (from training phase)
 * 
 * @param net: Network structure
 * @param weights: Weight matrices
 * @param biases: Bias vectors
 * @return: 0 on success, -1 on failure
 */
int neural_approximator_load_weights(NeuralApproximator* net,
                                     double** weights,
                                     double* biases);

/**
 * O(1) forward pass to approximate solution
 * 
 * @param net: Network structure
 * @param t: Time point
 * @param y0: Initial condition
 * @param params: ODE parameters
 * @param y_out: Output solution
 * @return: 0 on success, -1 on failure
 */
int neural_approximator_solve(NeuralApproximator* net,
                              double t,
                              const double* y0,
                              const double* params,
                              size_t num_params,
                              double* y_out);

/**
 * Free neural network approximator
 */
void neural_approximator_free(NeuralApproximator* net);

// ============================================================================
// Chebyshev Polynomial Approximator
// ============================================================================

/**
 * Initialize Chebyshev approximator
 * 
 * @param approx: Approximator structure
 * @param num_coefficients: Polynomial degree + 1
 * @param num_params: Number of parameter sets
 * @param param_values: Parameter values
 * @param time_scale: Time scaling factor
 * @return: 0 on success, -1 on failure
 */
int chebyshev_approximator_init(ChebyshevApproximator* approx,
                                size_t num_coefficients,
                                size_t num_params,
                                const double* param_values,
                                double time_scale);

/**
 * Load pre-computed Chebyshev coefficients
 * 
 * @param approx: Approximator structure
 * @param coefficients: Coefficient matrix [param_idx][coeff_idx]
 * @return: 0 on success, -1 on failure
 */
int chebyshev_approximator_load_coefficients(ChebyshevApproximator* approx,
                                            double** coefficients);

/**
 * O(1) Chebyshev evaluation
 * 
 * @param approx: Approximator structure
 * @param t: Time point
 * @param param_idx: Parameter index
 * @param y_out: Output solution
 * @return: 0 on success, -1 on failure
 */
int chebyshev_approximator_solve(ChebyshevApproximator* approx,
                                 double t,
                                 size_t param_idx,
                                 double* y_out);

/**
 * Free Chebyshev approximator
 */
void chebyshev_approximator_free(ChebyshevApproximator* approx);

// ============================================================================
// Reduced-Order Model Solver
// ============================================================================

/**
 * Initialize reduced-order model solver
 * 
 * @param rom: ROM structure
 * @param full_dim: Full system dimension
 * @param reduced_dim: Reduced dimension (r << n)
 * @param num_time_points: Number of pre-computed time points
 * @return: 0 on success, -1 on failure
 */
int reduced_order_init(ReducedOrderSolver* rom,
                      size_t full_dim,
                      size_t reduced_dim,
                      size_t num_time_points);

/**
 * Load projection matrix and reduced solutions (from offline POD/SVD)
 * 
 * @param rom: ROM structure
 * @param projection: Projection matrix [n × r]
 * @param reduced_solutions: Pre-computed solutions [time_idx][r]
 * @param time_points: Time points
 * @return: 0 on success, -1 on failure
 */
int reduced_order_load(ReducedOrderSolver* rom,
                      double** projection,
                      double** reduced_solutions,
                      const double* time_points);

/**
 * O(1) reduced-order solution
 * 
 * @param rom: ROM structure
 * @param t: Time point
 * @param y0: Initial condition
 * @param y_out: Output solution
 * @return: 0 on success, -1 on failure
 */
int reduced_order_solve(ReducedOrderSolver* rom,
                       double t,
                       const double* y0,
                       double* y_out);

/**
 * Free reduced-order solver
 */
void reduced_order_free(ReducedOrderSolver* rom);

// ============================================================================
// Hybrid O(1) Solver
// ============================================================================

/**
 * Initialize hybrid O(1) solver
 * 
 * @param solver: Solver structure
 * @param method: Approximation method to use
 * @return: 0 on success, -1 on failure
 */
int o1_solver_init(O1ApproximationSolver* solver, int method);

/**
 * O(1) solve with automatic method selection
 * 
 * @param solver: Solver structure
 * @param t: Time point
 * @param y0: Initial condition
 * @param params: ODE parameters
 * @param y_out: Output solution
 * @return: 0 on success, -1 on failure (falls back to traditional solver if enabled)
 */
int o1_solver_solve(O1ApproximationSolver* solver,
                    double t,
                    const double* y0,
                    const double* params,
                    size_t num_params,
                    double* y_out);

/**
 * Free hybrid solver
 */
void o1_solver_free(O1ApproximationSolver* solver);

#ifdef __cplusplus
}
#endif

#endif /* O1_APPROXIMATION_H */
