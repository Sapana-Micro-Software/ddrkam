/*
 * Cellular Automata Solvers for Differential Equations
 * Classical and Quantum Architectures
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef CELLULAR_AUTOMATA_H
#define CELLULAR_AUTOMATA_H

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
 * Cellular Automata Rule Type
 */
typedef enum {
    CA_ELEMENTARY,      // Elementary CA (1D)
    CA_GAME_OF_LIFE,    // Conway's Game of Life (2D)
    CA_TOTALISTIC,      // Totalistic CA
    CA_QUANTUM,         // Quantum CA (simulated)
    CA_CUSTOM           // Custom rule
} CARuleType;

/**
 * Quantum State (simulated)
 */
typedef struct {
    double real;        // Real part
    double imag;        // Imaginary part
    double probability; // |ψ|²
} QuantumState;

/**
 * Cellular Automata ODE Solver
 */
typedef struct {
    size_t state_dim;
    size_t grid_size;
    CARuleType rule_type;
    
    // Grid state
    double* grid_current;
    double* grid_next;
    size_t* neighborhood;
    
    // CA parameters
    uint32_t rule_number;  // For elementary CA
    double threshold;      // For threshold-based rules
    size_t neighborhood_size;
    
    // Quantum simulation (for quantum CA)
    QuantumState* quantum_grid;
    int use_quantum;
    
    // Performance
    size_t iterations;
    double convergence_threshold;
} CellularAutomataODESolver;

/**
 * Cellular Automata PDE Solver
 */
typedef struct {
    size_t spatial_dim;    // 1D, 2D, or 3D
    size_t* grid_size;      // Size per dimension
    CARuleType rule_type;
    
    // Multi-dimensional grid
    double* grid_current;
    double* grid_next;
    
    // CA parameters
    uint32_t rule_number;
    double threshold;
    size_t* neighborhood_size;
    
    // Quantum simulation
    QuantumState* quantum_grid;
    int use_quantum;
} CellularAutomataPDESolver;

/**
 * Initialize Cellular Automata ODE Solver
 */
int ca_ode_init(CellularAutomataODESolver* solver, size_t state_dim,
                size_t grid_size, CARuleType rule_type, uint32_t rule_number,
                int use_quantum);
void ca_ode_free(CellularAutomataODESolver* solver);
int ca_ode_solve(CellularAutomataODESolver* solver, ODEFunction f,
                double t0, double t_end, const double* y0,
                double h, void* params, double* y_out);

/**
 * Initialize Cellular Automata PDE Solver
 */
int ca_pde_init(CellularAutomataPDESolver* solver, size_t spatial_dim,
                const size_t* grid_size, CARuleType rule_type,
                uint32_t rule_number, int use_quantum);
void ca_pde_free(CellularAutomataPDESolver* solver);
int ca_pde_solve(CellularAutomataPDESolver* solver, double t0, double t_end,
                const double* u0, double h, double* u_out);

/**
 * Apply CA rule (elementary)
 */
void ca_apply_elementary_rule(double* grid_current, double* grid_next,
                              size_t grid_size, uint32_t rule_number);

/**
 * Apply CA rule (Game of Life)
 */
void ca_apply_game_of_life(double* grid_current, double* grid_next,
                          size_t width, size_t height);

/**
 * Apply quantum CA rule (simulated)
 */
void ca_apply_quantum_rule(QuantumState* grid_current, QuantumState* grid_next,
                          size_t grid_size, uint32_t rule_number);

#ifdef __cplusplus
}
#endif

#endif /* CELLULAR_AUTOMATA_H */
