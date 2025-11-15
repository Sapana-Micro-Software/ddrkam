/*
 * Map/Reduce Framework for Parallel ODE/PDE Solving
 * Designed for commodity hardware with redundancy
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef MAPREDUCE_SOLVERS_H
#define MAPREDUCE_SOLVERS_H

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
 * PDE function pointer type
 */
typedef void (*PDEFunction)(double t, double x, double y, const double* u,
                           double* dudt, double* dudx, double* dudy, void* params);

/**
 * Map function type: processes a chunk of data
 * @param key: Chunk identifier
 * @param value: Input data chunk
 * @param value_size: Size of input chunk
 * @param output: Output buffer
 * @param output_size: Size of output buffer
 * @param params: Additional parameters
 * @return: Number of output elements
 */
typedef size_t (*MapFunction)(const char* key, const double* value, size_t value_size,
                              double* output, size_t output_size, void* params);

/**
 * Reduce function type: aggregates mapped results
 * @param key: Result identifier
 * @param values: Array of mapped values
 * @param num_values: Number of values to reduce
 * @param output: Output buffer
 * @param params: Additional parameters
 * @return: Number of output elements
 */
typedef size_t (*ReduceFunction)(const char* key, const double* values, size_t num_values,
                                  double* output, void* params);

/**
 * Map/Reduce Configuration
 */
typedef struct {
    size_t num_mappers;          // Number of mapper nodes
    size_t num_reducers;         // Number of reducer nodes
    size_t chunk_size;           // Size of data chunks
    int enable_redundancy;       // Enable redundant computation for fault tolerance
    size_t redundancy_factor;    // Number of redundant copies (typically 3)
    int use_commodity_hardware;  // Optimize for commodity hardware
    double network_bandwidth;     // Network bandwidth (MB/s) for cost estimation
    double compute_cost_per_hour; // Cost per compute hour (for commodity hardware)
} MapReduceConfig;

/**
 * Map/Reduce Solver for ODEs
 * Implements Map/Reduce pattern for parallel ODE solving
 */
typedef struct {
    size_t state_dim;
    MapReduceConfig config;
    
    // Map phase: distribute state across mappers
    MapFunction map_func;
    double** map_inputs;          // Input chunks for each mapper
    double** map_outputs;        // Output from each mapper
    size_t* map_output_sizes;    // Size of each mapper output
    
    // Reduce phase: aggregate results
    ReduceFunction reduce_func;
    double* reduce_output;        // Final aggregated result
    
    // Fault tolerance
    double** redundant_outputs;  // Redundant copies for fault tolerance
    int* mapper_status;           // Status of each mapper (0=active, 1=failed)
    int* reducer_status;         // Status of each reducer
    
    // Performance metrics
    double map_time;              // Time spent in map phase
    double reduce_time;           // Time spent in reduce phase
    double shuffle_time;          // Time spent shuffling data
    size_t total_mappers_used;   // Total mappers used (including redundant)
    size_t total_reducers_used;  // Total reducers used
    
    // Commodity hardware optimization
    size_t* node_assignments;     // Assignment of chunks to nodes
    double* node_loads;          // Load on each node
} MapReduceODESolver;

/**
 * Map/Reduce Solver for PDEs
 * Extends Map/Reduce to spatial domains
 */
typedef struct {
    size_t spatial_dim;
    size_t* grid_size;
    MapReduceConfig config;
    
    // Spatial decomposition
    size_t* spatial_chunks;       // Spatial chunks per dimension
    double*** spatial_data;       // Spatial data chunks
    
    // Map/Reduce state
    MapFunction map_func;
    ReduceFunction reduce_func;
    double** map_outputs;
    double* reduce_output;
    
    // Performance metrics
    double map_time;
    double reduce_time;
    double shuffle_time;
} MapReducePDESolver;

/**
 * Initialize Map/Reduce ODE solver
 * 
 * Time Complexity: O(1) - constant initialization
 * Space Complexity: O(n + m + r) where n=state_dim, m=mappers, r=reducers
 * 
 * @param solver: Solver to initialize
 * @param state_dim: Dimension of ODE system
 * @param config: Map/Reduce configuration
 * @return: 0 on success, -1 on error
 */
int mapreduce_ode_init(MapReduceODESolver* solver, size_t state_dim,
                       const MapReduceConfig* config);

/**
 * Free Map/Reduce ODE solver
 */
void mapreduce_ode_free(MapReduceODESolver* solver);

/**
 * Solve ODE using Map/Reduce pattern
 * 
 * Time Complexity: O((n/m) * T_map + (m/r) * T_reduce + T_shuffle)
 *   where n=state_dim, m=mappers, r=reducers
 *   T_map = O(n/m) per mapper (parallel)
 *   T_reduce = O(m) per reducer (parallel)
 *   T_shuffle = O(m * log(m)) network communication
 * 
 * Overall: O(n/m + m/r + m*log(m)) with m processors
 * 
 * Space Complexity: O(n + m + r) for intermediate storage
 * 
 * @param solver: Map/Reduce solver
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param h: Step size
 * @param params: ODE parameters
 * @param y_out: Output state
 * @return: 0 on success, -1 on error
 */
int mapreduce_ode_solve(MapReduceODESolver* solver, ODEFunction f,
                        double t0, double t_end, const double* y0,
                        double h, void* params, double* y_out);

/**
 * Initialize Map/Reduce PDE solver
 */
int mapreduce_pde_init(MapReducePDESolver* solver, size_t spatial_dim,
                       const size_t* grid_size, const MapReduceConfig* config);

/**
 * Free Map/Reduce PDE solver
 */
void mapreduce_pde_free(MapReducePDESolver* solver);

/**
 * Solve PDE using Map/Reduce pattern
 * 
 * Time Complexity: O((G/m) * T_map + (m/r) * T_reduce + T_shuffle)
 *   where G=grid_size, m=mappers, r=reducers
 * 
 * @param solver: Map/Reduce PDE solver
 * @param f: PDE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param u0: Initial spatial state
 * @param h: Step size
 * @param params: PDE parameters
 * @param u_out: Output spatial state
 * @return: 0 on success, -1 on error
 */
int mapreduce_pde_solve(MapReducePDESolver* solver, PDEFunction f,
                        double t0, double t_end, const double* u0,
                        double h, void* params, double* u_out);

/**
 * Get performance metrics
 */
void mapreduce_get_metrics(const MapReduceODESolver* solver,
                          double* map_time, double* reduce_time,
                          double* shuffle_time, double* total_time);

/**
 * Estimate cost for commodity hardware
 * @param solver: Map/Reduce solver
 * @param compute_hours: Output estimated compute hours
 * @param network_cost: Output estimated network transfer cost
 * @return: Total estimated cost
 */
double mapreduce_estimate_cost(const MapReduceODESolver* solver,
                               double* compute_hours, double* network_cost);

#ifdef __cplusplus
}
#endif

#endif /* MAPREDUCE_SOLVERS_H */
