/*
 * Randomized Dynamic Programming for ODE Solving
 * Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "bayesian_ode_solvers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Xorshift RNG
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
    double u1 = uniform_random(state);
    double u2 = uniform_random(state);
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
}

// Sample states around current state
static void sample_states_around(RandomizedDPSolver* solver,
                                 const double* current_state,
                                 double** sampled_states) {
    for (size_t i = 0; i < solver->num_samples; i++) {
        for (size_t j = 0; j < solver->state_dim; j++) {
            double noise = gaussian_random(&solver->rng_state) * solver->sampling_radius;
            sampled_states[i][j] = current_state[j] + noise;
        }
    }
}

// Step forward using ODE
static void step_forward_ode(RandomizedDPSolver* solver,
                             double t,
                             const double* y_current,
                             double control,
                             double* y_next) {
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    if (!dydt) return;
    
    solver->ode_func(t, y_current, dydt, solver->ode_params);
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        y_next[i] = y_current[i] + control * dydt[i];
    }
    
    free(dydt);
}

// Find nearest sample to state
static size_t find_nearest_sample(RandomizedDPSolver* solver,
                                 const double* state) {
    double min_dist = DBL_MAX;
    size_t nearest = 0;
    
    for (size_t i = 0; i < solver->num_samples; i++) {
        double dist = 0.0;
        for (size_t j = 0; j < solver->state_dim; j++) {
            double diff = state[j] - solver->sampled_states[i][j];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    
    return nearest;
}

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
                      double ucb_constant) {
    if (!solver || !control_candidates || !ode_func || !cost_function) {
        return -1;
    }
    
    solver->state_dim = state_dim;
    solver->num_samples = num_samples;
    solver->num_controls = num_controls;
    solver->sampling_radius = sampling_radius;
    solver->ucb_constant = ucb_constant;
    solver->exploration_rate = 0.1;
    solver->use_ucb = 1;
    solver->ode_func = ode_func;
    solver->ode_params = ode_params;
    solver->cost_function = cost_function;
    solver->cost_params = cost_params;
    solver->step_count = 0;
    solver->total_samples = 0;
    solver->avg_step_time = 0.0;
    
    // Initialize RNG
    solver->rng_state = (uint32_t)time(NULL);
    
    // Allocate memory
    solver->sampled_states = (double**)malloc(num_samples * sizeof(double*));
    solver->state_weights = (double*)malloc(num_samples * sizeof(double));
    solver->value_estimates = (double*)malloc(num_samples * sizeof(double));
    solver->value_variance = (double*)malloc(num_samples * sizeof(double));
    solver->control_candidates = (double*)malloc(num_controls * sizeof(double));
    solver->control_counts = (size_t*)malloc(num_controls * sizeof(size_t));
    solver->best_control = (double*)malloc(num_samples * sizeof(double));
    solver->expected_value = (double*)malloc(sizeof(double));
    
    if (!solver->sampled_states || !solver->state_weights ||
        !solver->value_estimates || !solver->control_candidates ||
        !solver->control_counts || !solver->best_control) {
        randomized_dp_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < num_samples; i++) {
        solver->sampled_states[i] = (double*)malloc(state_dim * sizeof(double));
        if (!solver->sampled_states[i]) {
            randomized_dp_free(solver);
            return -1;
        }
        solver->state_weights[i] = 1.0 / num_samples;
        solver->value_estimates[i] = 0.0;
        solver->value_variance[i] = 0.0;
        solver->best_control[i] = control_candidates[0];
    }
    
    memcpy(solver->control_candidates, control_candidates,
           num_controls * sizeof(double));
    
    for (size_t i = 0; i < num_controls; i++) {
        solver->control_counts[i] = 1;  // Initialize to avoid division by zero
    }
    
    return 0;
}

void randomized_dp_free(RandomizedDPSolver* solver) {
    if (!solver) return;
    
    if (solver->sampled_states) {
        for (size_t i = 0; i < solver->num_samples; i++) {
            free(solver->sampled_states[i]);
        }
        free(solver->sampled_states);
    }
    
    free(solver->state_weights);
    free(solver->value_estimates);
    free(solver->value_variance);
    free(solver->control_candidates);
    free(solver->control_counts);
    free(solver->best_control);
    free(solver->expected_value);
    
    memset(solver, 0, sizeof(RandomizedDPSolver));
}

int randomized_dp_step(RandomizedDPSolver* solver,
                      double t,
                      const double* y_current,
                      double* y_next,
                      double* optimal_control) {
    if (!solver || !y_current || !y_next || !optimal_control) {
        return -1;
    }
    
    // Sample states around current state
    sample_states_around(solver, y_current, solver->sampled_states);
    solver->total_samples += solver->num_samples;
    
    // Estimate value for each control candidate
    double min_value = DBL_MAX;
    double best_u = solver->control_candidates[0];
    size_t best_control_idx = 0;
    
    for (size_t m = 0; m < solver->num_controls; m++) {
        double u = solver->control_candidates[m];
        
        // Estimate expected value via Monte Carlo
        double expected = 0.0;
        double variance_sum = 0.0;
        
        for (size_t i = 0; i < solver->num_samples; i++) {
            // Compute cost
            double cost = solver->cost_function(t, solver->sampled_states[i], u, solver->cost_params);
            
            // Step forward
            double* y_next_sample = (double*)malloc(solver->state_dim * sizeof(double));
            step_forward_ode(solver, t, solver->sampled_states[i], u, y_next_sample);
            
            // Lookup value at next state (simplified: use nearest sample)
            size_t nearest = find_nearest_sample(solver, y_next_sample);
            double next_value = solver->value_estimates[nearest];
            
            double total_value = cost + next_value;
            expected += total_value;
            variance_sum += total_value * total_value;
            
            free(y_next_sample);
        }
        
        expected /= solver->num_samples;
        double variance = (variance_sum / solver->num_samples) - (expected * expected);
        
        // UCB: add exploration bonus
        double ucb_value = expected;
        if (solver->use_ucb) {
            ucb_value -= solver->ucb_constant * sqrt(log(solver->step_count + 1) / solver->control_counts[m]);
        }
        
        // ε-greedy: random exploration
        if (!solver->use_ucb && uniform_random(&solver->rng_state) < solver->exploration_rate) {
            ucb_value = -DBL_MAX;  // Force exploration
        }
        
        if (ucb_value < min_value) {
            min_value = ucb_value;
            best_u = u;
            best_control_idx = m;
        }
        
        solver->control_counts[m]++;
    }
    
    // Apply optimal control
    step_forward_ode(solver, t, y_current, best_u, y_next);
    *optimal_control = best_u;
    
    // Update value estimates (simplified: update nearest samples)
    for (size_t i = 0; i < solver->num_samples; i++) {
        size_t nearest = find_nearest_sample(solver, y_next);
        solver->value_estimates[nearest] = min_value;
        solver->best_control[nearest] = best_u;
    }
    
    solver->step_count++;
    *solver->expected_value = min_value;
    
    return 0;
}

int randomized_dp_solve(RandomizedDPSolver* solver,
                        double t0,
                        double t_end,
                        const double* y0,
                        double** solution_path,
                        size_t num_steps,
                        double* controls) {
    if (!solver || !y0 || !solution_path) {
        return -1;
    }
    
    double time_step = (t_end - t0) / num_steps;
    double* y_current = (double*)malloc(solver->state_dim * sizeof(double));
    double* y_next = (double*)malloc(solver->state_dim * sizeof(double));
    
    if (!y_current || !y_next) {
        free(y_current);
        free(y_next);
        return -1;
    }
    
    memcpy(y_current, y0, solver->state_dim * sizeof(double));
    memcpy(solution_path[0], y0, solver->state_dim * sizeof(double));
    
    double t = t0;
    for (size_t step = 0; step < num_steps - 1; step++) {
        double optimal_control;
        
        if (randomized_dp_step(solver, t, y_current, y_next, &optimal_control) != 0) {
            free(y_current);
            free(y_next);
            return -1;
        }
        
        if (controls) {
            controls[step] = optimal_control;
        }
        
        memcpy(solution_path[step + 1], y_next, solver->state_dim * sizeof(double));
        memcpy(y_current, y_next, solver->state_dim * sizeof(double));
        t += time_step;
    }
    
    free(y_current);
    free(y_next);
    
    return 0;
}

int randomized_dp_get_value(RandomizedDPSolver* solver,
                          double t,
                          const double* y,
                          double* value_estimate,
                          double* value_variance) {
    if (!solver || !y || !value_estimate) {
        return -1;
    }
    
    // Find nearest sample and return its value estimate
    size_t nearest = find_nearest_sample(solver, y);
    *value_estimate = solver->value_estimates[nearest];
    
    if (value_variance) {
        *value_variance = solver->value_variance[nearest];
    }
    
    return 0;
}
