/*
 * Causal and Granger Causality Real-Time ODE Solvers
 * RK4 and Adams Methods with causal constraints and Granger causality analysis
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef CAUSAL_GRANGER_SOLVERS_H
#define CAUSAL_GRANGER_SOLVERS_H

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
 * Causal RK4 Solver
 * Real-time RK4 that only uses past information (strictly causal)
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // History buffer for causal operations (only past states)
    double** state_history;      // [history_size][state_dim]
    double* time_history;        // [history_size]
    size_t history_size;
    size_t history_count;
    size_t history_idx;
    
    // RK4 intermediate stages (causal: only use past)
    double* k1, *k2, *k3, *k4;
    double* y_temp;
    
    // Streaming buffer
    double* buffer;
    size_t buffer_size;
    size_t buffer_idx;
    
    // Performance metrics
    uint64_t total_steps;
    double avg_step_time;
    int strict_causality;        // 1 = strict causal (no future info), 0 = relaxed
} CausalRK4Solver;

/**
 * Causal Adams Method Solver
 * Real-time Adams that only uses past information
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // History for multi-step (causal: only past)
    double** state_history;      // [history_size][state_dim]
    double** derivative_history; // [history_size][state_dim] - f(t, y)
    double* time_history;        // [history_size]
    size_t history_size;
    size_t history_count;
    size_t history_idx;
    
    // Adams order (2, 3, or 4)
    size_t adams_order;
    
    // Streaming buffer
    double* buffer;
    size_t buffer_size;
    
    // Performance metrics
    uint64_t total_steps;
    double avg_step_time;
    int strict_causality;
} CausalAdamsSolver;

/**
 * Granger Causality Solver
 * Analyzes causal relationships between variables and adapts solving strategy
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // Granger causality analysis
    double** causality_matrix;   // [state_dim][state_dim] - Granger causality scores
    double** prediction_errors;   // [state_dim][state_dim] - Error when predicting i from j
    double* variable_importance; // [state_dim] - Importance of each variable
    size_t causality_window;      // Window size for causality analysis
    
    // History for causality computation
    double** state_history;      // [causality_window][state_dim]
    double* time_history;        // [causality_window]
    size_t history_count;
    size_t history_idx;
    
    // Adaptive solving based on causality
    int* causal_dependencies;    // [state_dim] - Which variables this depends on
    double* adaptive_weights;    // [state_dim] - Adaptive weights based on causality
    
    // Base solver (RK4 or Adams)
    enum {
        GRANGER_BASE_RK4 = 0,
        GRANGER_BASE_ADAMS = 1
    } base_method;
    
    void* base_solver;            // Pointer to CausalRK4Solver or CausalAdamsSolver
    
    // Performance metrics
    uint64_t total_steps;
    uint64_t causality_updates;
    double avg_step_time;
} GrangerCausalitySolver;

// ============================================================================
// Causal RK4 Solver
// ============================================================================

/**
 * Initialize causal RK4 solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @param history_size: Size of history buffer (for multi-step if needed)
 * @param strict_causality: 1 = strict causal (no future), 0 = relaxed
 * @return: 0 on success, -1 on failure
 */
int causal_rk4_init(CausalRK4Solver* solver,
                   size_t state_dim,
                   double step_size,
                   size_t history_size,
                   int strict_causality);

/**
 * Free causal RK4 solver
 */
void causal_rk4_free(CausalRK4Solver* solver);

/**
 * Causal RK4 step: only uses past information
 * O(1) per step with fixed state dimension
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function
 * @param t: Current time
 * @param y: Current state [state_dim] (input/output)
 * @param params: ODE parameters
 * @return: 0 on success, -1 on failure
 */
int causal_rk4_step(CausalRK4Solver* solver,
                   ODEFunction ode_func,
                   double t,
                   double* y,
                   void* params);

/**
 * Solve ODE using causal RK4
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial condition [state_dim]
 * @param params: ODE parameters
 * @param solution: Output solution [num_steps][state_dim]
 * @param num_steps: Number of steps
 * @return: 0 on success, -1 on failure
 */
int causal_rk4_solve(CausalRK4Solver* solver,
                    ODEFunction ode_func,
                    double t0,
                    double t_end,
                    const double* y0,
                    void* params,
                    double** solution,
                    size_t num_steps);

// ============================================================================
// Causal Adams Method Solver
// ============================================================================

/**
 * Initialize causal Adams solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @param adams_order: Order of Adams method (2, 3, or 4)
 * @param history_size: Size of history buffer
 * @param strict_causality: 1 = strict causal, 0 = relaxed
 * @return: 0 on success, -1 on failure
 */
int causal_adams_init(CausalAdamsSolver* solver,
                     size_t state_dim,
                     double step_size,
                     size_t adams_order,
                     size_t history_size,
                     int strict_causality);

/**
 * Free causal Adams solver
 */
void causal_adams_free(CausalAdamsSolver* solver);

/**
 * Causal Adams step: only uses past information
 * O(1) per step with fixed state dimension and order
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function
 * @param t: Current time
 * @param y: Current state [state_dim] (input/output)
 * @param params: ODE parameters
 * @return: 0 on success, -1 on failure
 */
int causal_adams_step(CausalAdamsSolver* solver,
                     ODEFunction ode_func,
                     double t,
                     double* y,
                     void* params);

/**
 * Solve ODE using causal Adams
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial condition [state_dim]
 * @param params: ODE parameters
 * @param solution: Output solution [num_steps][state_dim]
 * @param num_steps: Number of steps
 * @return: 0 on success, -1 on failure
 */
int causal_adams_solve(CausalAdamsSolver* solver,
                      ODEFunction ode_func,
                      double t0,
                      double t_end,
                      const double* y0,
                      void* params,
                      double** solution,
                      size_t num_steps);

// ============================================================================
// Granger Causality Solver
// ============================================================================

/**
 * Initialize Granger causality solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @param base_method: Base method (GRANGER_BASE_RK4 or GRANGER_BASE_ADAMS)
 * @param adams_order: Adams order (if using Adams, ignored for RK4)
 * @param causality_window: Window size for causality analysis
 * @return: 0 on success, -1 on failure
 */
int granger_causality_init(GrangerCausalitySolver* solver,
                          size_t state_dim,
                          double step_size,
                          int base_method,
                          size_t adams_order,
                          size_t causality_window);

/**
 * Free Granger causality solver
 */
void granger_causality_free(GrangerCausalitySolver* solver);

/**
 * Update Granger causality matrix from history
 * Computes which variables causally influence others
 * 
 * @param solver: Solver structure
 * @return: 0 on success, -1 on failure
 */
int granger_causality_update(GrangerCausalitySolver* solver);

/**
 * Granger causality step: uses causal relationships for adaptive solving
 * O(1) per step with fixed dimensions
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function
 * @param t: Current time
 * @param y: Current state [state_dim] (input/output)
 * @param params: ODE parameters
 * @return: 0 on success, -1 on failure
 */
int granger_causality_step(GrangerCausalitySolver* solver,
                          ODEFunction ode_func,
                          double t,
                          double* y,
                          void* params);

/**
 * Get Granger causality matrix
 * 
 * @param solver: Solver structure
 * @param causality_matrix: Output causality matrix [state_dim][state_dim]
 * @return: 0 on success, -1 on failure
 */
int granger_causality_get_matrix(GrangerCausalitySolver* solver,
                                 double** causality_matrix);

/**
 * Get variable importance scores
 * 
 * @param solver: Solver structure
 * @param importance: Output importance scores [state_dim]
 * @return: 0 on success, -1 on failure
 */
int granger_causality_get_importance(GrangerCausalitySolver* solver,
                                    double* importance);

/**
 * Solve ODE using Granger causality solver
 * 
 * @param solver: Solver structure
 * @param ode_func: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial condition [state_dim]
 * @param params: ODE parameters
 * @param solution: Output solution [num_steps][state_dim]
 * @param num_steps: Number of steps
 * @return: 0 on success, -1 on failure
 */
int granger_causality_solve(GrangerCausalitySolver* solver,
                           ODEFunction ode_func,
                           double t0,
                           double t_end,
                           const double* y0,
                           void* params,
                           double** solution,
                           size_t num_steps);

#ifdef __cplusplus
}
#endif

#endif /* CAUSAL_GRANGER_SOLVERS_H */
