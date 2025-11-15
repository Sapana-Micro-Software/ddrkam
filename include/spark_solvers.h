/*
 * Apache Spark Framework for Parallel ODE/PDE Solving
 * Designed for commodity hardware with fault tolerance
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef SPARK_SOLVERS_H
#define SPARK_SOLVERS_H

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
 * Spark Configuration
 */
typedef struct {
    size_t num_executors;        // Number of Spark executors
    size_t cores_per_executor;   // CPU cores per executor
    size_t memory_per_executor;  // Memory per executor (MB)
    size_t num_partitions;       // Number of RDD partitions
    int enable_caching;          // Enable RDD caching
    int enable_checkpointing;    // Enable checkpointing for fault tolerance
    double checkpoint_interval;  // Checkpoint interval (seconds)
    int use_commodity_hardware;  // Optimize for commodity hardware
    double network_bandwidth;    // Network bandwidth (MB/s)
    double compute_cost_per_hour; // Cost per compute hour
    int enable_dynamic_allocation; // Enable dynamic resource allocation
} SparkConfig;

/**
 * Spark RDD (Resilient Distributed Dataset) representation
 */
typedef struct {
    double* data;                // Data array
    size_t data_size;            // Size of data
    size_t num_partitions;       // Number of partitions
    size_t* partition_sizes;    // Size of each partition
    double** partition_data;    // Data for each partition
    int is_cached;              // Whether RDD is cached
    int is_checkpointed;        // Whether RDD is checkpointed
} SparkRDD;

/**
 * Spark Solver for ODEs
 * Implements Spark RDD operations for parallel ODE solving
 */
typedef struct {
    size_t state_dim;
    SparkConfig config;
    
    // Spark RDDs for state
    SparkRDD* state_rdd;         // Current state RDD
    SparkRDD* derivative_rdd;   // Derivative RDD
    SparkRDD* result_rdd;       // Result RDD
    
    // Spark operations
    double** executor_results;   // Results from each executor
    size_t* executor_loads;      // Load on each executor
    
    // Fault tolerance
    SparkRDD** checkpoint_rdds; // Checkpointed RDDs
    size_t num_checkpoints;
    int* executor_status;       // Status of each executor
    
    // Performance metrics
    double map_time;             // Time in map operations
    double reduce_time;          // Time in reduce operations
    double shuffle_time;         // Time in shuffle operations
    double cache_hit_rate;       // Cache hit rate (0.0 to 1.0)
    size_t num_tasks;            // Total number of tasks
    size_t num_completed_tasks;  // Completed tasks
    
    // Commodity hardware optimization
    double* executor_utilization; // Utilization of each executor
    double total_cost;           // Total estimated cost
} SparkODESolver;

/**
 * Spark Solver for PDEs
 * Extends Spark to spatial domains with RDD partitioning
 */
typedef struct {
    size_t spatial_dim;
    size_t* grid_size;
    SparkConfig config;
    
    // Spatial RDDs
    SparkRDD* spatial_rdd;       // Spatial state RDD
    SparkRDD* spatial_derivative_rdd; // Spatial derivative RDD
    
    // Spatial partitioning
    size_t* spatial_partitions;   // Partitions per dimension
    
    // Performance metrics
    double map_time;
    double reduce_time;
    double shuffle_time;
    double cache_hit_rate;
} SparkPDESolver;

/**
 * Initialize Spark ODE solver
 * 
 * Time Complexity: O(1) - constant initialization
 * Space Complexity: O(n + e) where n=state_dim, e=executors
 * 
 * @param solver: Solver to initialize
 * @param state_dim: Dimension of ODE system
 * @param config: Spark configuration
 * @return: 0 on success, -1 on error
 */
int spark_ode_init(SparkODESolver* solver, size_t state_dim,
                   const SparkConfig* config);

/**
 * Free Spark ODE solver
 */
void spark_ode_free(SparkODESolver* solver);

/**
 * Solve ODE using Spark RDD operations
 * 
 * Time Complexity: O((n/p) * T_map + (p/e) * T_reduce + T_shuffle)
 *   where n=state_dim, p=partitions, e=executors
 *   T_map = O(n/p) per partition (parallel)
 *   T_reduce = O(p) per executor (parallel)
 *   T_shuffle = O(p * log(p)) network communication
 * 
 * With caching: O(n/p + p/e + p*log(p)) for first iteration,
 *               O(n/p + p/e) for subsequent iterations (cache hits)
 * 
 * Space Complexity: O(n + p + e) for RDD storage and caching
 * 
 * @param solver: Spark solver
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param h: Step size
 * @param params: ODE parameters
 * @param y_out: Output state
 * @return: 0 on success, -1 on error
 */
int spark_ode_solve(SparkODESolver* solver, ODEFunction f,
                    double t0, double t_end, const double* y0,
                    double h, void* params, double* y_out);

/**
 * Initialize Spark PDE solver
 */
int spark_pde_init(SparkPDESolver* solver, size_t spatial_dim,
                  const size_t* grid_size, const SparkConfig* config);

/**
 * Free Spark PDE solver
 */
void spark_pde_free(SparkPDESolver* solver);

/**
 * Solve PDE using Spark RDD operations
 * 
 * Time Complexity: O((G/p) * T_map + (p/e) * T_reduce + T_shuffle)
 *   where G=grid_size, p=partitions, e=executors
 * 
 * @param solver: Spark PDE solver
 * @param f: PDE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param u0: Initial spatial state
 * @param h: Step size
 * @param params: PDE parameters
 * @param u_out: Output spatial state
 * @return: 0 on success, -1 on error
 */
int spark_pde_solve(SparkPDESolver* solver, PDEFunction f,
                    double t0, double t_end, const double* u0,
                    double h, void* params, double* u_out);

/**
 * Get Spark performance metrics
 */
void spark_get_metrics(const SparkODESolver* solver,
                      double* map_time, double* reduce_time,
                      double* shuffle_time, double* cache_hit_rate,
                      double* total_time);

/**
 * Estimate cost for commodity hardware
 * @param solver: Spark solver
 * @param compute_hours: Output estimated compute hours
 * @param network_cost: Output estimated network transfer cost
 * @param storage_cost: Output estimated storage cost
 * @return: Total estimated cost
 */
double spark_estimate_cost(const SparkODESolver* solver,
                          double* compute_hours, double* network_cost,
                          double* storage_cost);

/**
 * Create Spark RDD from data
 */
SparkRDD* spark_create_rdd(const double* data, size_t data_size,
                          size_t num_partitions);

/**
 * Free Spark RDD
 */
void spark_free_rdd(SparkRDD* rdd);

/**
 * Cache Spark RDD
 */
int spark_cache_rdd(SparkRDD* rdd);

/**
 * Checkpoint Spark RDD
 */
int spark_checkpoint_rdd(SparkRDD* rdd);

#ifdef __cplusplus
}
#endif

#endif /* SPARK_SOLVERS_H */
