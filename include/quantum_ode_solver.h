/*
 * Quantum ODE Solver with Nonlinear DE Methods
 * Post-real-time performance for future prediction
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef QUANTUM_ODE_SOLVER_H
#define QUANTUM_ODE_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Nonlinear ODE function (can include nonlinear terms, delays, etc.)
 */
typedef void (*NonlinearODEFunction)(double t, const double* y, double* dydt, 
                                     const double* y_history, size_t history_size,
                                     void* params);

/**
 * Quantum state representation
 * Uses quantum-inspired superposition of classical states
 */
typedef struct {
    double* amplitude_real;      // Real part of amplitude [num_states]
    double* amplitude_imag;      // Imaginary part of amplitude [num_states]
    double* probabilities;       // |amplitude|² [num_states]
    size_t num_states;           // Number of quantum states
    double normalization;        // Normalization constant
} QuantumState;

/**
 * Quantum-inspired optimization parameters
 */
typedef struct {
    double temperature;          // Quantum annealing temperature
    double tunneling_strength;   // Quantum tunneling parameter
    double coherence_time;       // Quantum coherence time
    size_t num_iterations;       // Number of optimization iterations
    double convergence_threshold;
} QuantumParams;

/**
 * Quantum Nonlinear ODE Solver
 * Uses quantum-inspired methods for nonlinear ODE solving and future prediction
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // Quantum state representation
    QuantumState* quantum_states;  // [num_quantum_states]
    size_t num_quantum_states;
    
    // Nonlinear ODE solver
    NonlinearODEFunction nonlinear_ode_func;
    void* ode_params;
    
    // History for nonlinear terms (delays, memory effects)
    double** state_history;      // [history_size][state_dim]
    double* time_history;         // [history_size]
    size_t history_size;
    size_t history_count;
    size_t history_idx;
    
    // Quantum optimization
    QuantumParams quantum_params;
    double* energy_landscape;     // Energy landscape for optimization
    double* gradient;             // Quantum gradient
    
    // Future prediction
    double* predicted_future;    // [prediction_horizon][state_dim]
    size_t prediction_horizon;
    double prediction_confidence; // Confidence in predictions
    
    // Post-real-time refinement
    int use_post_realtime;        // Enable post-real-time refinement
    double* refined_solution;     // Refined solution after real-time
    size_t refinement_iterations;
    
    // Performance metrics
    uint64_t total_steps;
    uint64_t quantum_operations;
    double avg_step_time;
    double prediction_time;
} QuantumODESolver;

/**
 * Quantum Variational Solver
 * Variational quantum algorithm for ODE solving
 */
typedef struct {
    size_t state_dim;
    double* variational_params;   // [num_params] - quantum circuit parameters
    size_t num_variational_params;
    
    // Variational ansatz
    void (*ansatz_function)(const double* params, size_t num_params,
                           const double* y, size_t state_dim,
                           double* output, void* user_data);
    void* ansatz_data;
    
    // Cost function for optimization
    double (*cost_function)(const double* params, size_t num_params,
                           const double* y, size_t state_dim,
                           void* user_data);
    void* cost_data;
    
    // Optimization
    double learning_rate;
    size_t max_iterations;
    double convergence_threshold;
    
    // Results
    double* optimal_params;
    double optimal_cost;
} QuantumVariationalSolver;

// ============================================================================
// Quantum State Operations
// ============================================================================

/**
 * Initialize quantum state
 * 
 * @param state: Quantum state structure
 * @param num_states: Number of quantum states
 * @return: 0 on success, -1 on failure
 */
int quantum_state_init(QuantumState* state, size_t num_states);

/**
 * Free quantum state
 */
void quantum_state_free(QuantumState* state);

/**
 * Normalize quantum state (ensure |ψ|² = 1)
 * 
 * @param state: Quantum state
 * @return: 0 on success, -1 on failure
 */
int quantum_state_normalize(QuantumState* state);

/**
 * Measure quantum state (collapse to classical state)
 * 
 * @param state: Quantum state
 * @param measured_state: Output measured state index
 * @return: 0 on success, -1 on failure
 */
int quantum_state_measure(QuantumState* state, size_t* measured_state);

/**
 * Apply quantum gate (unitary transformation)
 * 
 * @param state: Quantum state
 * @param gate_matrix: Unitary matrix [num_states][num_states]
 * @return: 0 on success, -1 on failure
 */
int quantum_state_apply_gate(QuantumState* state, double** gate_matrix);

// ============================================================================
// Quantum ODE Solver
// ============================================================================

/**
 * Initialize quantum ODE solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @param num_quantum_states: Number of quantum states for superposition
 * @param history_size: History size for nonlinear terms
 * @param prediction_horizon: Number of future steps to predict
 * @param nonlinear_ode_func: Nonlinear ODE function
 * @param ode_params: ODE parameters
 * @param quantum_params: Quantum parameters
 * @return: 0 on success, -1 on failure
 */
int quantum_ode_init(QuantumODESolver* solver,
                    size_t state_dim,
                    double step_size,
                    size_t num_quantum_states,
                    size_t history_size,
                    size_t prediction_horizon,
                    NonlinearODEFunction nonlinear_ode_func,
                    void* ode_params,
                    const QuantumParams* quantum_params);

/**
 * Free quantum ODE solver
 */
void quantum_ode_free(QuantumODESolver* solver);

/**
 * Quantum ODE step: solve using quantum-inspired optimization
 * Post-real-time: can take longer than real-time for better accuracy
 * 
 * @param solver: Solver structure
 * @param t: Current time
 * @param y: Current state [state_dim] (input/output)
 * @return: 0 on success, -1 on failure
 */
int quantum_ode_step(QuantumODESolver* solver,
                    double t,
                    double* y);

/**
 * Predict future states using quantum superposition
 * 
 * @param solver: Solver structure
 * @param t: Current time
 * @param y_current: Current state [state_dim]
 * @param future_states: Output future states [prediction_horizon][state_dim]
 * @param confidence: Output prediction confidence
 * @return: 0 on success, -1 on failure
 */
int quantum_ode_predict_future(QuantumODESolver* solver,
                               double t,
                               const double* y_current,
                               double** future_states,
                               double* confidence);

/**
 * Post-real-time refinement: improve solution after real-time
 * 
 * @param solver: Solver structure
 * @param t: Time point
 * @param y_initial: Initial solution [state_dim]
 * @param y_refined: Output refined solution [state_dim]
 * @param max_iterations: Maximum refinement iterations
 * @return: 0 on success, -1 on failure
 */
int quantum_ode_refine(QuantumODESolver* solver,
                      double t,
                      const double* y_initial,
                      double* y_refined,
                      size_t max_iterations);

/**
 * Solve ODE using quantum methods
 * 
 * @param solver: Solver structure
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial condition [state_dim]
 * @param solution: Output solution [num_steps][state_dim]
 * @param num_steps: Number of steps
 * @return: 0 on success, -1 on failure
 */
int quantum_ode_solve(QuantumODESolver* solver,
                     double t0,
                     double t_end,
                     const double* y0,
                     double** solution,
                     size_t num_steps);

// ============================================================================
// Quantum Variational Solver
// ============================================================================

/**
 * Initialize quantum variational solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param num_variational_params: Number of variational parameters
 * @param ansatz_function: Variational ansatz function
 * @param ansatz_data: Data for ansatz
 * @param cost_function: Cost function for optimization
 * @param cost_data: Data for cost function
 * @param learning_rate: Learning rate for optimization
 * @return: 0 on success, -1 on failure
 */
int quantum_variational_init(QuantumVariationalSolver* solver,
                            size_t state_dim,
                            size_t num_variational_params,
                            void (*ansatz_function)(const double* params, size_t num_params,
                                                   const double* y, size_t state_dim,
                                                   double* output, void* user_data),
                            void* ansatz_data,
                            double (*cost_function)(const double* params, size_t num_params,
                                                    const double* y, size_t state_dim,
                                                    void* user_data),
                            void* cost_data,
                            double learning_rate);

/**
 * Free quantum variational solver
 */
void quantum_variational_free(QuantumVariationalSolver* solver);

/**
 * Optimize variational parameters
 * 
 * @param solver: Solver structure
 * @param initial_params: Initial parameters [num_params]
 * @param optimal_params: Output optimal parameters [num_params]
 * @param optimal_cost: Output optimal cost
 * @return: 0 on success, -1 on failure
 */
int quantum_variational_optimize(QuantumVariationalSolver* solver,
                                const double* initial_params,
                                double* optimal_params,
                                double* optimal_cost);

/**
 * Solve ODE using variational quantum algorithm
 * 
 * @param solver: Solver structure
 * @param t: Time point
 * @param y: State [state_dim] (input/output)
 * @return: 0 on success, -1 on failure
 */
int quantum_variational_solve(QuantumVariationalSolver* solver,
                              double t,
                              double* y);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create quantum annealing schedule
 * 
 * @param temperature: Current temperature
 * @param initial_temp: Initial temperature
 * @param final_temp: Final temperature
 * @param iteration: Current iteration
 * @param max_iterations: Maximum iterations
 * @return: Scheduled temperature
 */
double quantum_annealing_schedule(double initial_temp,
                                 double final_temp,
                                 size_t iteration,
                                 size_t max_iterations);

/**
 * Quantum tunneling probability
 * 
 * @param energy_barrier: Energy barrier height
 * @param tunneling_strength: Tunneling strength parameter
 * @return: Tunneling probability
 */
double quantum_tunneling_probability(double energy_barrier,
                                    double tunneling_strength);

/**
 * Quantum superposition of classical solutions
 * 
 * @param solutions: Array of classical solutions [num_solutions][state_dim]
 * @param weights: Weights for each solution [num_solutions]
 * @param num_solutions: Number of solutions
 * @param state_dim: State dimension
 * @param superposed: Output superposed solution [state_dim]
 * @return: 0 on success, -1 on failure
 */
int quantum_superposition(const double** solutions,
                         const double* weights,
                         size_t num_solutions,
                         size_t state_dim,
                         double* superposed);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_ODE_SOLVER_H */
