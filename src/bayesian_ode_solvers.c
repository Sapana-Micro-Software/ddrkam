/*
 * Real-Time Bayesian ODE Solvers with Dynamic Programming
 * Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "bayesian_ode_solvers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Simple random number generator (Xorshift)
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double uniform_random(uint32_t* state) {
    return ((double)xorshift32(state)) / UINT32_MAX;
}

static double gaussian_random(uint32_t* state) {
    // Box-Muller transform
    double u1 = uniform_random(state);
    double u2 = uniform_random(state);
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
}

// ============================================================================
// Forward-Backward Solver
// ============================================================================

int forward_backward_init(ForwardBackwardSolver* solver,
                         size_t state_space_size,
                         const double* state_values,
                         double** transition_matrix,
                         const double* prior,
                         double observation_noise_variance) {
    if (!solver || !state_values || !transition_matrix || !prior) {
        return -1;
    }
    
    solver->state_space_size = state_space_size;
    solver->observation_noise_variance = observation_noise_variance;
    solver->current_time = 0.0;
    solver->current_step = 0;
    solver->total_steps = 0;
    solver->avg_step_time = 0.0;
    
    // Allocate memory
    solver->state_values = (double*)malloc(state_space_size * sizeof(double));
    solver->alpha = (double*)malloc(state_space_size * sizeof(double));
    solver->beta = (double*)malloc(state_space_size * sizeof(double));
    solver->posterior = (double*)malloc(state_space_size * sizeof(double));
    solver->prior = (double*)malloc(state_space_size * sizeof(double));
    solver->mean = (double*)malloc(sizeof(double));
    solver->variance = (double*)malloc(sizeof(double));
    solver->std_dev = (double*)malloc(sizeof(double));
    
    solver->transition_matrix = (double**)malloc(state_space_size * sizeof(double*));
    for (size_t i = 0; i < state_space_size; i++) {
        solver->transition_matrix[i] = (double*)malloc(state_space_size * sizeof(double));
        memcpy(solver->transition_matrix[i], transition_matrix[i],
               state_space_size * sizeof(double));
    }
    
    if (!solver->state_values || !solver->alpha || !solver->beta ||
        !solver->posterior || !solver->prior || !solver->transition_matrix) {
        forward_backward_free(solver);
        return -1;
    }
    
    // Copy state values and prior
    memcpy(solver->state_values, state_values, state_space_size * sizeof(double));
    memcpy(solver->prior, prior, state_space_size * sizeof(double));
    
    // Initialize alpha with prior
    memcpy(solver->alpha, prior, state_space_size * sizeof(double));
    normalize_probabilities(solver->alpha, state_space_size);
    
    // Initialize beta to uniform
    for (size_t i = 0; i < state_space_size; i++) {
        solver->beta[i] = 1.0 / state_space_size;
    }
    
    return 0;
}

void forward_backward_free(ForwardBackwardSolver* solver) {
    if (!solver) return;
    
    if (solver->transition_matrix) {
        for (size_t i = 0; i < solver->state_space_size; i++) {
            free(solver->transition_matrix[i]);
        }
        free(solver->transition_matrix);
    }
    
    free(solver->state_values);
    free(solver->alpha);
    free(solver->beta);
    free(solver->posterior);
    free(solver->prior);
    free(solver->mean);
    free(solver->variance);
    free(solver->std_dev);
    
    memset(solver, 0, sizeof(ForwardBackwardSolver));
}

int forward_backward_step(ForwardBackwardSolver* solver, double observation) {
    if (!solver) return -1;
    
    // Compute observation likelihood for each state
    double* observation_likelihood = (double*)malloc(solver->state_space_size * sizeof(double));
    if (!observation_likelihood) return -1;
    
    for (size_t i = 0; i < solver->state_space_size; i++) {
        double diff = observation - solver->state_values[i];
        observation_likelihood[i] = gaussian_observation_likelihood(
            observation, solver->state_values[i], solver->observation_noise_variance);
    }
    
    // Forward update: α(t) = Σ α(t-1) × transition × observation
    double* alpha_new = (double*)malloc(solver->state_space_size * sizeof(double));
    if (!alpha_new) {
        free(observation_likelihood);
        return -1;
    }
    
    for (size_t j = 0; j < solver->state_space_size; j++) {
        alpha_new[j] = 0.0;
        for (size_t i = 0; i < solver->state_space_size; i++) {
            alpha_new[j] += solver->alpha[i] * 
                           solver->transition_matrix[i][j] *
                           observation_likelihood[j];
        }
    }
    
    normalize_probabilities(alpha_new, solver->state_space_size);
    memcpy(solver->alpha, alpha_new, solver->state_space_size * sizeof(double));
    
    free(alpha_new);
    free(observation_likelihood);
    
    solver->current_step++;
    solver->total_steps++;
    
    return 0;
}

int forward_backward_compute_posterior(ForwardBackwardSolver* solver) {
    if (!solver) return -1;
    
    // Posterior ∝ α × β
    for (size_t i = 0; i < solver->state_space_size; i++) {
        solver->posterior[i] = solver->alpha[i] * solver->beta[i];
    }
    
    normalize_probabilities(solver->posterior, solver->state_space_size);
    
    return 0;
}

int forward_backward_get_statistics(ForwardBackwardSolver* solver,
                                   double* y_mean,
                                   double* y_variance,
                                   double* full_posterior) {
    if (!solver || !y_mean || !y_variance) return -1;
    
    // Compute mean
    *y_mean = 0.0;
    for (size_t i = 0; i < solver->state_space_size; i++) {
        *y_mean += solver->posterior[i] * solver->state_values[i];
    }
    
    // Compute variance
    *y_variance = 0.0;
    for (size_t i = 0; i < solver->state_space_size; i++) {
        double diff = solver->state_values[i] - *y_mean;
        *y_variance += solver->posterior[i] * diff * diff;
    }
    
    *solver->mean = *y_mean;
    *solver->variance = *y_variance;
    *solver->std_dev = sqrt(*y_variance);
    
    if (full_posterior) {
        memcpy(full_posterior, solver->posterior,
               solver->state_space_size * sizeof(double));
    }
    
    return 0;
}

// ============================================================================
// Viterbi Solver
// ============================================================================

int viterbi_init(ViterbiSolver* solver,
                size_t state_space_size,
                const double* state_values,
                double** transition_matrix,
                const double* prior,
                double observation_noise_variance) {
    if (!solver || !state_values || !transition_matrix || !prior) {
        return -1;
    }
    
    solver->state_space_size = state_space_size;
    solver->observation_noise_variance = observation_noise_variance;
    solver->current_time = 0.0;
    solver->current_step = 0;
    solver->total_steps = 0;
    solver->map_probability = 0.0;
    
    // Allocate memory
    solver->state_values = (double*)malloc(state_space_size * sizeof(double));
    solver->viterbi = (double*)malloc(state_space_size * sizeof(double));
    solver->backpointers = (size_t*)malloc(state_space_size * sizeof(size_t));
    solver->prior = (double*)malloc(state_space_size * sizeof(double));
    
    solver->transition_matrix = (double**)malloc(state_space_size * sizeof(double*));
    for (size_t i = 0; i < state_space_size; i++) {
        solver->transition_matrix[i] = (double*)malloc(state_space_size * sizeof(double));
        memcpy(solver->transition_matrix[i], transition_matrix[i],
               state_space_size * sizeof(double));
    }
    
    if (!solver->state_values || !solver->viterbi || !solver->backpointers ||
        !solver->prior || !solver->transition_matrix) {
        viterbi_free(solver);
        return -1;
    }
    
    memcpy(solver->state_values, state_values, state_space_size * sizeof(double));
    memcpy(solver->prior, prior, state_space_size * sizeof(double));
    
    // Initialize Viterbi table with prior
    for (size_t i = 0; i < state_space_size; i++) {
        solver->viterbi[i] = log(solver->prior[i] + 1e-10);
        solver->backpointers[i] = 0;
    }
    
    return 0;
}

void viterbi_free(ViterbiSolver* solver) {
    if (!solver) return;
    
    if (solver->transition_matrix) {
        for (size_t i = 0; i < solver->state_space_size; i++) {
            free(solver->transition_matrix[i]);
        }
        free(solver->transition_matrix);
    }
    
    free(solver->state_values);
    free(solver->viterbi);
    free(solver->backpointers);
    free(solver->prior);
    free(solver->map_path);
    
    memset(solver, 0, sizeof(ViterbiSolver));
}

int viterbi_step(ViterbiSolver* solver, double observation) {
    if (!solver) return -1;
    
    // Compute observation likelihood
    double* obs_likelihood = (double*)malloc(solver->state_space_size * sizeof(double));
    if (!obs_likelihood) return -1;
    
    for (size_t i = 0; i < solver->state_space_size; i++) {
        obs_likelihood[i] = gaussian_observation_likelihood(
            observation, solver->state_values[i], solver->observation_noise_variance);
    }
    
    // Viterbi update: V(t, s) = max_{s_prev} [V(t-1, s_prev) + log(transition) + log(obs)]
    double* viterbi_new = (double*)malloc(solver->state_space_size * sizeof(double));
    size_t* backpointers_new = (size_t*)malloc(solver->state_space_size * sizeof(size_t));
    
    if (!viterbi_new || !backpointers_new) {
        free(obs_likelihood);
        free(viterbi_new);
        free(backpointers_new);
        return -1;
    }
    
    for (size_t j = 0; j < solver->state_space_size; j++) {
        double max_value = -DBL_MAX;
        size_t best_prev = 0;
        
        for (size_t i = 0; i < solver->state_space_size; i++) {
            double value = solver->viterbi[i] +
                          log(solver->transition_matrix[i][j] + 1e-10) +
                          log(obs_likelihood[j] + 1e-10);
            
            if (value > max_value) {
                max_value = value;
                best_prev = i;
            }
        }
        
        viterbi_new[j] = max_value;
        backpointers_new[j] = best_prev;
    }
    
    memcpy(solver->viterbi, viterbi_new, solver->state_space_size * sizeof(double));
    memcpy(solver->backpointers, backpointers_new, solver->state_space_size * sizeof(size_t));
    
    free(obs_likelihood);
    free(viterbi_new);
    free(backpointers_new);
    
    solver->current_step++;
    solver->total_steps++;
    
    return 0;
}

int viterbi_get_map(ViterbiSolver* solver, double* y_map, double* map_probability) {
    if (!solver || !y_map) return -1;
    
    // Find state with maximum Viterbi value
    double max_value = -DBL_MAX;
    size_t best_state = 0;
    
    for (size_t i = 0; i < solver->state_space_size; i++) {
        if (solver->viterbi[i] > max_value) {
            max_value = solver->viterbi[i];
            best_state = i;
        }
    }
    
    *y_map = solver->state_values[best_state];
    
    if (map_probability) {
        *map_probability = exp(max_value);
    }
    solver->map_probability = exp(max_value);
    
    return 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

double gaussian_observation_likelihood(double observation,
                                      double predicted,
                                      double noise_variance) {
    double diff = observation - predicted;
    return exp(-0.5 * diff * diff / noise_variance) / sqrt(2.0 * M_PI * noise_variance);
}

void normalize_probabilities(double* probabilities, size_t n) {
    if (!probabilities || n == 0) return;
    
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += probabilities[i];
    }
    
    if (sum > 1e-10) {
        for (size_t i = 0; i < n; i++) {
            probabilities[i] /= sum;
        }
    } else {
        // Uniform if sum is too small
        double uniform = 1.0 / n;
        for (size_t i = 0; i < n; i++) {
            probabilities[i] = uniform;
        }
    }
}

// Placeholder implementations for other functions
// (Full implementations would be in separate files or extended here)

int particle_filter_init(ParticleFilterSolver* solver,
                        size_t num_particles,
                        size_t state_dim,
                        TransitionModel transition_model,
                        ObservationLikelihood observation_likelihood,
                        void* model_params) {
    // Implementation placeholder
    return 0;
}

void particle_filter_free(ParticleFilterSolver* solver) {
    // Implementation placeholder
}

int realtime_bayesian_init(RealTimeBayesianSolver* solver,
                           BayesianMode mode,
                           size_t state_space_size,
                           const double* state_values,
                           double** transition_matrix,
                           const double* prior,
                           double observation_noise_variance) {
    // Implementation placeholder
    return 0;
}

void realtime_bayesian_free(RealTimeBayesianSolver* solver) {
    // Implementation placeholder
}

double compute_effective_sample_size(const double* weights, size_t num_particles) {
    if (!weights || num_particles == 0) return 0.0;
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < num_particles; i++) {
        sum_sq += weights[i] * weights[i];
    }
    
    return 1.0 / (sum_sq + 1e-10);
}
