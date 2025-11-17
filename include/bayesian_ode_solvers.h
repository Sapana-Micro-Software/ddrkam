/*
 * Real-Time Bayesian ODE Solvers with Dynamic Programming
 * Probabilistic and Exact (MAP) solutions in O(1) time
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef BAYESIAN_ODE_SOLVERS_H
#define BAYESIAN_ODE_SOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * Solver mode: probabilistic, exact (MAP), or both
 */
typedef enum {
    BAYESIAN_MODE_PROBABILISTIC = 0,  // Full posterior distribution
    BAYESIAN_MODE_EXACT = 1,          // MAP estimate (Viterbi)
    BAYESIAN_MODE_HYBRID = 2          // Both probabilistic and exact
} BayesianMode;

/**
 * Forward-Backward probabilistic solver
 * Computes full posterior p(y(t) | observations)
 */
typedef struct {
    // Pre-computed (offline)
    double** transition_matrix;    // [S][S] - p(y(t+1)=j | y(t)=i)
    double* state_values;          // [S] - discretized state values
    size_t state_space_size;       // S - fixed constant
    
    // Forward probabilities: α(t) = p(y(t) | observations[0:t])
    double* alpha;
    
    // Backward probabilities: β(t) = p(observations[t+1:T] | y(t))
    double* beta;
    
    // Posterior: p(y(t) | all observations) ∝ α(t) × β(t)
    double* posterior;
    
    // Prior distribution
    double* prior;
    
    // Statistics
    double* mean;                  // Mean estimate E[y(t)]
    double* variance;              // Variance estimate Var[y(t)]
    double* std_dev;                // Standard deviation
    
    // Current time and state
    double current_time;
    size_t current_step;
    
    // Observation model parameters
    double observation_noise_variance;
    
    // Performance metrics
    uint64_t total_steps;
    double avg_step_time;
} ForwardBackwardSolver;

/**
 * Viterbi exact (MAP) solver
 * Finds most likely path argmax p(y(t) | observations)
 */
typedef struct {
    // Pre-computed (offline)
    double** transition_matrix;    // [S][S] - transition probabilities
    double* state_values;          // [S] - discretized states
    size_t state_space_size;       // S - fixed constant
    
    // Viterbi table: V(t, s) = max prob ending at state s at time t
    double* viterbi;
    
    // Backpointers for path reconstruction
    size_t* backpointers;
    
    // MAP path
    size_t* map_path;              // Most likely state sequence
    double map_probability;        // Probability of MAP path
    
    // Prior
    double* prior;
    
    // Current state
    double current_time;
    size_t current_step;
    
    // Observation model
    double observation_noise_variance;
    
    // Performance
    uint64_t total_steps;
    double avg_step_time;
} ViterbiSolver;

/**
 * Particle filter solver (approximate probabilistic)
 * Uses Monte Carlo sampling for nonlinear/non-Gaussian cases
 */
typedef struct {
    // Particles: {y₁(t), y₂(t), ..., yₙ(t)}
    double** particles;             // [N][state_dim]
    double* weights;                // [N] - particle weights
    size_t num_particles;           // N - fixed constant
    
    // State dimension
    size_t state_dim;
    
    // Transition and observation models
    void (*transition_model)(double t, const double* y_prev, double* y_next, void* params);
    double (*observation_likelihood)(double observation, const double* y, void* params);
    void* model_params;
    
    // Resampling
    int use_systematic_resampling;
    double effective_sample_size_threshold;  // Resample if ESS < threshold
    
    // Statistics
    double* mean;                   // Mean estimate
    double* variance;               // Variance estimate
    double* covariance;             // Covariance matrix [state_dim × state_dim]
    
    // Current state
    double current_time;
    size_t current_step;
    
    // Performance
    uint64_t total_steps;
    uint64_t resampling_count;
    double avg_step_time;
} ParticleFilterSolver;

/**
 * Hybrid Bayesian solver
 * Combines probabilistic and exact methods
 */
typedef struct {
    BayesianMode mode;
    
    ForwardBackwardSolver* forward_backward;
    ViterbiSolver* viterbi;
    ParticleFilterSolver* particle_filter;
    
    // Unified output
    double* y_probabilistic;       // Probabilistic estimate (mean)
    double* y_exact;                // Exact estimate (MAP)
    double* y_variance;            // Uncertainty (variance)
    
    // Selection: which method to use
    int use_forward_backward;
    int use_viterbi;
    int use_particle_filter;
} HybridBayesianSolver;

/**
 * Real-time Bayesian ODE solver
 * O(1) per-step with pre-computed transitions
 */
typedef struct {
    BayesianMode mode;
    
    // Pre-computed transition matrix (offline)
    double** transition_matrix;     // [S][S]
    double* state_values;           // [S]
    size_t state_space_size;        // S - fixed
    
    // Online state
    double* alpha;                  // Forward probabilities
    double* beta;                   // Backward probabilities (if needed)
    double* posterior;             // Current posterior
    double* viterbi;                // Viterbi table (if exact mode)
    size_t* backpointers;          // Path reconstruction (if exact mode)
    
    // Outputs
    double* y_mean;                 // Mean estimate
    double* y_variance;             // Variance estimate
    double* y_map;                  // MAP estimate
    
    // Observation model
    double observation_noise_variance;
    
    // Current state
    double current_time;
    size_t current_step;
    
    // Performance
    uint64_t total_steps;
    double avg_step_time;
} RealTimeBayesianSolver;

// ============================================================================
// Forward-Backward Solver (Probabilistic)
// ============================================================================

/**
 * Initialize forward-backward solver
 * 
 * @param solver: Solver structure
 * @param state_space_size: Size of discretized state space (fixed constant)
 * @param state_values: Discretized state values [S]
 * @param transition_matrix: Pre-computed transition matrix [S][S]
 * @param prior: Prior distribution [S]
 * @param observation_noise_variance: Variance of observation noise
 * @return: 0 on success, -1 on failure
 */
int forward_backward_init(ForwardBackwardSolver* solver,
                         size_t state_space_size,
                         const double* state_values,
                         double** transition_matrix,
                         const double* prior,
                         double observation_noise_variance);

/**
 * Free forward-backward solver
 */
void forward_backward_free(ForwardBackwardSolver* solver);

/**
 * Forward step: update α(t) = p(y(t) | observations[0:t])
 * O(1) with fixed state space size
 * 
 * @param solver: Solver structure
 * @param observation: Current observation
 * @return: 0 on success, -1 on failure
 */
int forward_backward_step(ForwardBackwardSolver* solver, double observation);

/**
 * Backward step: update β(t) = p(observations[t+1:T] | y(t))
 * O(1) with fixed state space size
 * 
 * @param solver: Solver structure
 * @param observation: Observation at next time step
 * @return: 0 on success, -1 on failure
 */
int forward_backward_backward_step(ForwardBackwardSolver* solver, double observation);

/**
 * Compute posterior: p(y(t) | all observations) ∝ α(t) × β(t)
 * 
 * @param solver: Solver structure
 * @return: 0 on success, -1 on failure
 */
int forward_backward_compute_posterior(ForwardBackwardSolver* solver);

/**
 * Get statistics: mean, variance, full posterior
 * 
 * @param solver: Solver structure
 * @param y_mean: Output mean estimate
 * @param y_variance: Output variance estimate
 * @param full_posterior: Output full posterior [S] (optional, can be NULL)
 * @return: 0 on success, -1 on failure
 */
int forward_backward_get_statistics(ForwardBackwardSolver* solver,
                                   double* y_mean,
                                   double* y_variance,
                                   double* full_posterior);

// ============================================================================
// Viterbi Solver (Exact/MAP)
// ============================================================================

/**
 * Initialize Viterbi solver
 * 
 * @param solver: Solver structure
 * @param state_space_size: Size of discretized state space
 * @param state_values: Discretized state values [S]
 * @param transition_matrix: Pre-computed transition matrix [S][S]
 * @param prior: Prior distribution [S]
 * @param observation_noise_variance: Variance of observation noise
 * @return: 0 on success, -1 on failure
 */
int viterbi_init(ViterbiSolver* solver,
                size_t state_space_size,
                const double* state_values,
                double** transition_matrix,
                const double* prior,
                double observation_noise_variance);

/**
 * Free Viterbi solver
 */
void viterbi_free(ViterbiSolver* solver);

/**
 * Viterbi step: update V(t, s) and find most likely path
 * O(1) with fixed state space size
 * 
 * @param solver: Solver structure
 * @param observation: Current observation
 * @return: 0 on success, -1 on failure
 */
int viterbi_step(ViterbiSolver* solver, double observation);

/**
 * Reconstruct MAP path from backpointers
 * 
 * @param solver: Solver structure
 * @param map_path: Output MAP path [T] (state indices)
 * @param path_length: Length of path
 * @return: 0 on success, -1 on failure
 */
int viterbi_reconstruct_path(ViterbiSolver* solver,
                            size_t* map_path,
                            size_t path_length);

/**
 * Get MAP estimate
 * 
 * @param solver: Solver structure
 * @param y_map: Output MAP estimate
 * @param map_probability: Output probability of MAP path (optional, can be NULL)
 * @return: 0 on success, -1 on failure
 */
int viterbi_get_map(ViterbiSolver* solver,
                   double* y_map,
                   double* map_probability);

// ============================================================================
// Particle Filter Solver (Approximate Probabilistic)
// ============================================================================

/**
 * Transition model function type
 * 
 * @param t: Current time
 * @param y_prev: Previous state [state_dim]
 * @param y_next: Next state [state_dim] (output)
 * @param params: Model parameters
 */
typedef void (*TransitionModel)(double t, const double* y_prev, double* y_next, void* params);

/**
 * Observation likelihood function type
 * 
 * @param observation: Observed value
 * @param y: Current state [state_dim]
 * @param params: Model parameters
 * @return: Likelihood p(observation | y)
 */
typedef double (*ObservationLikelihood)(double observation, const double* y, void* params);

/**
 * Initialize particle filter solver
 * 
 * @param solver: Solver structure
 * @param num_particles: Number of particles (fixed constant for O(1))
 * @param state_dim: State dimension
 * @param transition_model: Transition model function
 * @param observation_likelihood: Observation likelihood function
 * @param model_params: Parameters for models
 * @return: 0 on success, -1 on failure
 */
int particle_filter_init(ParticleFilterSolver* solver,
                        size_t num_particles,
                        size_t state_dim,
                        TransitionModel transition_model,
                        ObservationLikelihood observation_likelihood,
                        void* model_params);

/**
 * Free particle filter solver
 */
void particle_filter_free(ParticleFilterSolver* solver);

/**
 * Particle filter step: propagate, update weights, resample
 * O(1) with fixed number of particles
 * 
 * @param solver: Solver structure
 * @param t: Current time
 * @param observation: Current observation
 * @return: 0 on success, -1 on failure
 */
int particle_filter_step(ParticleFilterSolver* solver,
                         double t,
                         double observation);

/**
 * Get statistics from particles
 * 
 * @param solver: Solver structure
 * @param y_mean: Output mean estimate [state_dim]
 * @param y_variance: Output variance estimate [state_dim]
 * @param y_covariance: Output covariance matrix [state_dim × state_dim] (optional)
 * @return: 0 on success, -1 on failure
 */
int particle_filter_get_statistics(ParticleFilterSolver* solver,
                                  double* y_mean,
                                  double* y_variance,
                                  double* y_covariance);

// ============================================================================
// Real-Time Bayesian Solver (O(1) per step)
// ============================================================================

/**
 * Initialize real-time Bayesian solver
 * 
 * @param solver: Solver structure
 * @param mode: Solver mode (probabilistic, exact, or hybrid)
 * @param state_space_size: Size of discretized state space (fixed)
 * @param state_values: Discretized state values [S]
 * @param transition_matrix: Pre-computed transition matrix [S][S]
 * @param prior: Prior distribution [S]
 * @param observation_noise_variance: Variance of observation noise
 * @return: 0 on success, -1 on failure
 */
int realtime_bayesian_init(RealTimeBayesianSolver* solver,
                           BayesianMode mode,
                           size_t state_space_size,
                           const double* state_values,
                           double** transition_matrix,
                           const double* prior,
                           double observation_noise_variance);

/**
 * Free real-time Bayesian solver
 */
void realtime_bayesian_free(RealTimeBayesianSolver* solver);

/**
 * O(1) real-time step: update probabilities and compute estimates
 * 
 * @param solver: Solver structure
 * @param t: Current time
 * @param observation: Current observation
 * @param y_out: Output solution (mean if probabilistic, MAP if exact)
 * @return: 0 on success, -1 on failure
 */
int realtime_bayesian_step(RealTimeBayesianSolver* solver,
                           double t,
                           double observation,
                           double* y_out);

/**
 * Get full statistics (both probabilistic and exact if hybrid mode)
 * 
 * @param solver: Solver structure
 * @param y_mean: Output mean estimate (probabilistic)
 * @param y_variance: Output variance estimate
 * @param y_map: Output MAP estimate (exact)
 * @return: 0 on success, -1 on failure
 */
int realtime_bayesian_get_statistics(RealTimeBayesianSolver* solver,
                                    double* y_mean,
                                    double* y_variance,
                                    double* y_map);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Pre-compute transition matrix from ODE
 * 
 * @param ode_func: ODE function dy/dt = f(t, y, params)
 * @param state_values: Discretized state values [S]
 * @param state_space_size: Size of state space
 * @param time_step: Time step Δt
 * @param params: ODE parameters
 * @param transition_matrix: Output transition matrix [S][S]
 * @return: 0 on success, -1 on failure
 */
int precompute_transition_matrix(void (*ode_func)(double t, const double* y, double* dydt, void* params),
                                const double* state_values,
                                size_t state_space_size,
                                double time_step,
                                void* params,
                                double** transition_matrix);

/**
 * Compute observation likelihood (Gaussian)
 * 
 * @param observation: Observed value
 * @param predicted: Predicted value
 * @param noise_variance: Variance of observation noise
 * @return: Likelihood p(observation | predicted)
 */
double gaussian_observation_likelihood(double observation,
                                        double predicted,
                                        double noise_variance);

/**
 * Normalize probability distribution
 * 
 * @param probabilities: Probability vector [n] (modified in-place)
 * @param n: Size of vector
 */
void normalize_probabilities(double* probabilities, size_t n);

/**
 * Compute effective sample size (for particle filters)
 * 
 * @param weights: Particle weights [N]
 * @param num_particles: Number of particles
 * @return: Effective sample size
 */
double compute_effective_sample_size(const double* weights, size_t num_particles);

/**
 * Systematic resampling for particle filter
 * 
 * @param particles: Particles [N][state_dim]
 * @param weights: Weights [N]
 * @param num_particles: Number of particles
 * @param state_dim: State dimension
 */
void systematic_resample(double** particles,
                       double* weights,
                       size_t num_particles,
                       size_t state_dim);

// ============================================================================
// Randomized Dynamic Programming Solver
// ============================================================================

/**
 * Cost function type for randomized DP
 * 
 * @param t: Current time
 * @param y: Current state [state_dim]
 * @param u: Control/action (e.g., step size)
 * @param params: User parameters
 * @return: Cost L(y, u, t)
 */
typedef double (*CostFunction)(double t, const double* y, double u, void* params);

/**
 * Randomized Dynamic Programming solver
 * Uses Monte Carlo sampling for efficient value function estimation
 */
typedef struct {
    // State space sampling
    double** sampled_states;      // [N][state_dim] - random state samples
    double* state_weights;        // [N] - importance weights
    size_t num_samples;           // N - fixed constant for O(1)
    size_t state_dim;             // State dimension
    
    // Value function estimates
    double* value_estimates;       // [N] - V(t, y_i) for each sample
    double* value_variance;        // [N] - variance of estimates
    
    // Control/action space
    double* control_candidates;   // [M] - candidate step sizes/methods
    size_t num_controls;          // M - fixed constant
    size_t* control_counts;        // [M] - visit counts for UCB
    
    // Sampling parameters
    double sampling_radius;       // Radius for state sampling
    double exploration_rate;      // ε for ε-greedy
    double ucb_constant;          // c for UCB exploration
    int use_ucb;                  // Use UCB vs ε-greedy
    
    // ODE function
    void (*ode_func)(double t, const double* y, double* dydt, void* params);
    void* ode_params;
    
    // Cost function
    CostFunction cost_function;
    void* cost_params;
    
    // Optimal policy
    double* best_control;         // [N] - optimal control for each sample
    double* expected_value;       // Expected value estimate
    
    // Statistics
    uint64_t step_count;
    uint64_t total_samples;
    double avg_step_time;
    
    // Random number generator state
    uint32_t rng_state;
} RandomizedDPSolver;

/**
 * Initialize randomized DP solver
 * 
 * @param solver: Solver structure
 * @param state_dim: State dimension
 * @param num_samples: Number of samples (fixed constant for O(1))
 * @param num_controls: Number of control candidates
 * @param control_candidates: Control candidates [M] (e.g., step sizes)
 * @param ode_func: ODE function
 * @param ode_params: ODE parameters
 * @param cost_function: Cost function
 * @param cost_params: Cost function parameters
 * @param sampling_radius: Radius for state sampling
 * @param ucb_constant: UCB exploration constant
 * @return: 0 on success, -1 on failure
 */
int randomized_dp_init(RandomizedDPSolver* solver,
                      size_t state_dim,
                      size_t num_samples,
                      size_t num_controls,
                      const double* control_candidates,
                      void (*ode_func)(double t, const double* y, double* dydt, void* params),
                      void* ode_params,
                      CostFunction cost_function,
                      void* cost_params,
                      double sampling_radius,
                      double ucb_constant);

/**
 * Free randomized DP solver
 */
void randomized_dp_free(RandomizedDPSolver* solver);

/**
 * O(1) randomized DP step: estimate value and choose optimal control
 * 
 * @param solver: Solver structure
 * @param t: Current time
 * @param y_current: Current state [state_dim]
 * @param y_next: Next state [state_dim] (output)
 * @param optimal_control: Optimal control/step size (output)
 * @return: 0 on success, -1 on failure
 */
int randomized_dp_step(RandomizedDPSolver* solver,
                      double t,
                      const double* y_current,
                      double* y_next,
                      double* optimal_control);

/**
 * Solve ODE using randomized DP (backward induction)
 * 
 * @param solver: Solver structure
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial condition [state_dim]
 * @param solution_path: Output solution path [num_steps][state_dim]
 * @param num_steps: Number of time steps
 * @param controls: Output optimal controls [num_steps] (optional)
 * @return: 0 on success, -1 on failure
 */
int randomized_dp_solve(RandomizedDPSolver* solver,
                        double t0,
                        double t_end,
                        const double* y0,
                        double** solution_path,
                        size_t num_steps,
                        double* controls);

/**
 * Get value function estimate at state
 * 
 * @param solver: Solver structure
 * @param t: Time
 * @param y: State [state_dim]
 * @param value_estimate: Output value estimate
 * @param value_variance: Output variance estimate (optional)
 * @return: 0 on success, -1 on failure
 */
int randomized_dp_get_value(RandomizedDPSolver* solver,
                          double t,
                          const double* y,
                          double* value_estimate,
                          double* value_variance);

#ifdef __cplusplus
}
#endif

#endif /* BAYESIAN_ODE_SOLVERS_H */
