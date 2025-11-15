/*
 * Petri Net Solvers for Differential Equations
 * Classical and Quantum Architectures
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef PETRI_NET_H
#define PETRI_NET_H

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
 * Place in Petri Net
 */
typedef struct {
    size_t id;
    double tokens;          // Number of tokens (continuous)
    double capacity;        // Maximum capacity
    double* marking_history; // History of markings
} PetriPlace;

/**
 * Transition in Petri Net
 */
typedef struct {
    size_t id;
    double firing_rate;     // Firing rate (for continuous Petri nets)
    size_t* input_places;   // Input place indices
    size_t* output_places;  // Output place indices
    size_t num_inputs;
    size_t num_outputs;
    double* weights;        // Arc weights
    int enabled;            // Whether transition is enabled
} PetriTransition;

/**
 * Quantum Place (simulated)
 */
typedef struct {
    size_t id;
    QuantumState state;     // Quantum state
    double probability;     // Measurement probability
} QuantumPetriPlace;

/**
 * Petri Net ODE Solver
 */
typedef struct {
    size_t num_places;
    size_t num_transitions;
    
    PetriPlace* places;
    PetriTransition* transitions;
    
    // Firing sequence
    size_t* firing_sequence;
    size_t sequence_length;
    
    // Quantum simulation
    QuantumPetriPlace* quantum_places;
    int use_quantum;
    
    // Performance
    size_t steps;
    double time;
} PetriNetODESolver;

/**
 * Petri Net PDE Solver
 */
typedef struct {
    size_t num_places;
    size_t num_transitions;
    size_t spatial_dim;
    size_t* grid_size;
    
    PetriPlace** spatial_places;  // Places distributed in space
    PetriTransition** spatial_transitions;
    
    // Quantum simulation
    QuantumPetriPlace** quantum_spatial_places;
    int use_quantum;
} PetriNetPDESolver;

/**
 * Initialize Petri Net ODE Solver
 */
int petri_net_ode_init(PetriNetODESolver* solver, size_t num_places,
                       size_t num_transitions, int use_quantum);
void petri_net_ode_free(PetriNetODESolver* solver);
int petri_net_ode_solve(PetriNetODESolver* solver, ODEFunction f,
                        double t0, double t_end, const double* y0,
                        double h, void* params, double* y_out);

/**
 * Initialize Petri Net PDE Solver
 */
int petri_net_pde_init(PetriNetPDESolver* solver, size_t num_places,
                       size_t num_transitions, size_t spatial_dim,
                       const size_t* grid_size, int use_quantum);
void petri_net_pde_free(PetriNetPDESolver* solver);
int petri_net_pde_solve(PetriNetPDESolver* solver, double t0, double t_end,
                        const double* u0, double h, double* u_out);

/**
 * Fire transition in Petri net
 */
int petri_fire_transition(PetriNetODESolver* solver, size_t transition_id);

/**
 * Check if transition is enabled
 */
int petri_transition_enabled(PetriNetODESolver* solver, size_t transition_id);

/**
 * Quantum measurement (simulated)
 */
double petri_quantum_measure(QuantumPetriPlace* place);

#ifdef __cplusplus
}
#endif

#endif /* PETRI_NET_H */
