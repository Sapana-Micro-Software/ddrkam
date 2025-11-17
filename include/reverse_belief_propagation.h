/*
 * Reverse Belief Propagation with Lossless Tracing
 * Backwards uncertainty propagation for ODE solving
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef REVERSE_BELIEF_PROPAGATION_H
#define REVERSE_BELIEF_PROPAGATION_H

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
 * Belief structure: represents uncertainty/confidence in state
 */
typedef struct {
    double* mean;                   // Mean estimate [state_dim]
    double** covariance;            // Covariance matrix [state_dim][state_dim]
    double* confidence;             // Confidence per dimension [state_dim]
    size_t state_dim;
    double timestamp;
} Belief;

/**
 * Lossless trace entry: stores complete state information
 */
typedef struct {
    double time;
    double* state;                  // [state_dim] - exact state
    double* derivative;             // [state_dim] - exact derivative
    double** jacobian;              // [state_dim][state_dim] - ∂f/∂y
    double* sensitivity;            // [state_dim] - sensitivity to parameters
    Belief belief;                  // Belief at this time
    size_t state_dim;
    uint64_t trace_id;              // Unique trace identifier
} LosslessTraceEntry;

/**
 * Reverse Belief Propagation Solver
 * Propagates beliefs backwards in time with lossless tracing
 */
typedef struct {
    size_t state_dim;
    double current_time;
    double* current_state;
    double step_size;
    
    // Forward pass: store lossless trace
    LosslessTraceEntry* trace;      // [trace_size] - complete forward trace
    size_t trace_size;
    size_t trace_count;
    size_t trace_capacity;
    
    // Reverse pass: belief propagation
    Belief* belief_history;         // [trace_size] - beliefs at each time
    double** belief_transition;     // [state_dim][state_dim] - belief transition matrix
    
    // Lossless tracing parameters
    int store_jacobian;             // Store full Jacobian (lossless)
    int store_sensitivity;          // Store parameter sensitivity
    int use_exact_arithmetic;       // Use exact arithmetic where possible
    
    // ODE function
    ODEFunction ode_func;
    void* ode_params;
    
    // Jacobian computation
    void (*jacobian_func)(double t, const double* y, double** jacobian, void* params);
    void* jacobian_params;
    
    // Performance metrics
    uint64_t forward_steps;
    uint64_t reverse_steps;
    uint64_t trace_operations;
    double avg_forward_time;
    double avg_reverse_time;
} ReverseBeliefPropagationSolver;

/**
 * Initialize reverse belief propagation solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @param trace_capacity: Maximum number of trace entries
 * @param ode_func: ODE function
 * @param ode_params: ODE parameters
 * @param jacobian_func: Jacobian function (optional, NULL for numerical)
 * @param jacobian_params: Jacobian parameters
 * @param store_jacobian: 1 = store full Jacobian (lossless), 0 = approximate
 * @param store_sensitivity: 1 = store parameter sensitivity
 * @return: 0 on success, -1 on failure
 */
int reverse_belief_init(ReverseBeliefPropagationSolver* solver,
                       size_t state_dim,
                       double step_size,
                       size_t trace_capacity,
                       ODEFunction ode_func,
                       void* ode_params,
                       void (*jacobian_func)(double t, const double* y, double** jacobian, void* params),
                       void* jacobian_params,
                       int store_jacobian,
                       int store_sensitivity);

/**
 * Free reverse belief propagation solver
 */
void reverse_belief_free(ReverseBeliefPropagationSolver* solver);

/**
 * Forward step: solve forward and store lossless trace
 * 
 * @param solver: Solver structure
 * @param t: Current time
 * @param y: Current state [state_dim] (input/output)
 * @param belief: Current belief (input/output)
 * @return: 0 on success, -1 on failure
 */
int reverse_belief_forward_step(ReverseBeliefPropagationSolver* solver,
                               double t,
                               double* y,
                               Belief* belief);

/**
 * Reverse step: propagate belief backwards using lossless trace
 * 
 * @param solver: Solver structure
 * @param t: Time point (going backwards)
 * @param y: State [state_dim] (output, updated from trace)
 * @param belief: Belief (output, propagated backwards)
 * @return: 0 on success, -1 on failure
 */
int reverse_belief_reverse_step(ReverseBeliefPropagationSolver* solver,
                               double t,
                               double* y,
                               Belief* belief);

/**
 * Solve forward and build lossless trace
 * 
 * @param solver: Solver structure
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial condition [state_dim]
 * @param initial_belief: Initial belief
 * @return: 0 on success, -1 on failure
 */
int reverse_belief_forward_solve(ReverseBeliefPropagationSolver* solver,
                                 double t0,
                                 double t_end,
                                 const double* y0,
                                 const Belief* initial_belief);

/**
 * Reverse solve: propagate beliefs backwards using lossless trace
 * 
 * @param solver: Solver structure
 * @param t_start: Starting time (going backwards from)
 * @param t_end: Ending time (going backwards to)
 * @param final_belief: Final belief (at t_start)
 * @param solution: Output solution [num_steps][state_dim]
 * @param beliefs: Output beliefs [num_steps]
 * @param num_steps: Number of steps
 * @return: 0 on success, -1 on failure
 */
int reverse_belief_reverse_solve(ReverseBeliefPropagationSolver* solver,
                                 double t_start,
                                 double t_end,
                                 const Belief* final_belief,
                                 double** solution,
                                 Belief* beliefs,
                                 size_t num_steps);

/**
 * Get lossless trace entry at time
 * 
 * @param solver: Solver structure
 * @param t: Time point
 * @param trace_entry: Output trace entry
 * @return: 0 on success, -1 if not found
 */
int reverse_belief_get_trace(ReverseBeliefPropagationSolver* solver,
                             double t,
                             LosslessTraceEntry* trace_entry);

/**
 * Smooth solution: combine forward and reverse passes
 * 
 * @param solver: Solver structure
 * @param t: Time point
 * @param y_forward: Forward solution [state_dim]
 * @param belief_forward: Forward belief
 * @param y_reverse: Reverse solution [state_dim]
 * @param belief_reverse: Reverse belief
 * @param y_smoothed: Output smoothed solution [state_dim]
 * @param belief_smoothed: Output smoothed belief
 * @return: 0 on success, -1 on failure
 */
int reverse_belief_smooth(ReverseBeliefPropagationSolver* solver,
                         double t,
                         const double* y_forward,
                         const Belief* belief_forward,
                         const double* y_reverse,
                         const Belief* belief_reverse,
                         double* y_smoothed,
                         Belief* belief_smoothed);

// ============================================================================
// Belief Operations
// ============================================================================

/**
 * Initialize belief
 * 
 * @param belief: Belief structure
 * @param state_dim: State dimension
 * @param mean: Mean estimate [state_dim]
 * @param covariance: Covariance matrix [state_dim][state_dim]
 * @return: 0 on success, -1 on failure
 */
int belief_init(Belief* belief,
               size_t state_dim,
               const double* mean,
               double** covariance);

/**
 * Free belief
 */
void belief_free(Belief* belief);

/**
 * Copy belief
 * 
 * @param dest: Destination belief
 * @param src: Source belief
 * @return: 0 on success, -1 on failure
 */
int belief_copy(Belief* dest, const Belief* src);

/**
 * Propagate belief forward through ODE
 * 
 * @param belief: Belief (input/output)
 * @param jacobian: Jacobian matrix [state_dim][state_dim]
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @return: 0 on success, -1 on failure
 */
int belief_propagate_forward(Belief* belief,
                            double** jacobian,
                            size_t state_dim,
                            double step_size);

/**
 * Propagate belief backward (reverse)
 * 
 * @param belief: Belief (input/output)
 * @param jacobian: Jacobian matrix [state_dim][state_dim]
 * @param state_dim: State dimension
 * @param step_size: Step size
 * @return: 0 on success, -1 on failure
 */
int belief_propagate_backward(Belief* belief,
                             double** jacobian,
                             size_t state_dim,
                             double step_size);

/**
 * Combine two beliefs (for smoothing)
 * 
 * @param belief1: First belief
 * @param belief2: Second belief
 * @param belief_combined: Output combined belief
 * @return: 0 on success, -1 on failure
 */
int belief_combine(const Belief* belief1,
                  const Belief* belief2,
                  Belief* belief_combined);

// ============================================================================
// Lossless Trace Operations
// ============================================================================

/**
 * Initialize lossless trace entry
 * 
 * @param entry: Trace entry
 * @param state_dim: State dimension
 * @param time: Time point
 * @return: 0 on success, -1 on failure
 */
int trace_entry_init(LosslessTraceEntry* entry,
                    size_t state_dim,
                    double time);

/**
 * Free lossless trace entry
 */
void trace_entry_free(LosslessTraceEntry* entry);

/**
 * Store complete state in trace (lossless)
 * 
 * @param entry: Trace entry
 * @param state: State [state_dim]
 * @param derivative: Derivative [state_dim]
 * @param jacobian: Jacobian [state_dim][state_dim] (optional)
 * @param sensitivity: Sensitivity [state_dim] (optional)
 * @return: 0 on success, -1 on failure
 */
int trace_entry_store(LosslessTraceEntry* entry,
                     const double* state,
                     const double* derivative,
                     double** jacobian,
                     const double* sensitivity);

/**
 * Retrieve state from trace (lossless reconstruction)
 * 
 * @param entry: Trace entry
 * @param state: Output state [state_dim]
 * @param derivative: Output derivative [state_dim] (optional)
 * @return: 0 on success, -1 on failure
 */
int trace_entry_retrieve(const LosslessTraceEntry* entry,
                        double* state,
                        double* derivative);

#ifdef __cplusplus
}
#endif

#endif /* REVERSE_BELIEF_PROPAGATION_H */
