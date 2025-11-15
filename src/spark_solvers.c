/*
 * Apache Spark Framework Implementation for ODE/PDE Solving
 * Designed for commodity hardware with fault tolerance
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "spark_solvers.h"
#include "rk3.h"
#include "euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// Create Spark RDD from data
SparkRDD* spark_create_rdd(const double* data, size_t data_size, size_t num_partitions) {
    if (!data || data_size == 0 || num_partitions == 0) {
        return NULL;
    }
    
    SparkRDD* rdd = (SparkRDD*)malloc(sizeof(SparkRDD));
    if (!rdd) {
        return NULL;
    }
    
    memset(rdd, 0, sizeof(SparkRDD));
    rdd->data_size = data_size;
    rdd->num_partitions = num_partitions;
    rdd->is_cached = 0;
    rdd->is_checkpointed = 0;
    
    // Allocate partition arrays
    rdd->partition_sizes = (size_t*)malloc(num_partitions * sizeof(size_t));
    rdd->partition_data = (double**)malloc(num_partitions * sizeof(double*));
    
    if (!rdd->partition_sizes || !rdd->partition_data) {
        spark_free_rdd(rdd);
        return NULL;
    }
    
    // Partition data
    size_t partition_size = (data_size + num_partitions - 1) / num_partitions;
    for (size_t i = 0; i < num_partitions; i++) {
        size_t start_idx = i * partition_size;
        size_t end_idx = (start_idx + partition_size < data_size) ?
                        start_idx + partition_size : data_size;
        size_t local_size = end_idx - start_idx;
        
        rdd->partition_sizes[i] = local_size;
        rdd->partition_data[i] = (double*)malloc(local_size * sizeof(double));
        
        if (!rdd->partition_data[i]) {
            spark_free_rdd(rdd);
            return NULL;
        }
        
        memcpy(rdd->partition_data[i], &data[start_idx], local_size * sizeof(double));
    }
    
    return rdd;
}

void spark_free_rdd(SparkRDD* rdd) {
    if (!rdd) return;
    
    if (rdd->partition_data) {
        for (size_t i = 0; i < rdd->num_partitions; i++) {
            if (rdd->partition_data[i]) free(rdd->partition_data[i]);
        }
        free(rdd->partition_data);
    }
    
    if (rdd->partition_sizes) free(rdd->partition_sizes);
    if (rdd->data) free(rdd->data);
    
    free(rdd);
}

int spark_cache_rdd(SparkRDD* rdd) {
    if (!rdd) return -1;
    rdd->is_cached = 1;
    return 0;
}

int spark_checkpoint_rdd(SparkRDD* rdd) {
    if (!rdd) return -1;
    rdd->is_checkpointed = 1;
    return 0;
}

int spark_ode_init(SparkODESolver* solver, size_t state_dim, const SparkConfig* config) {
    if (!solver || state_dim == 0 || !config) {
        return -1;
    }
    
    memset(solver, 0, sizeof(SparkODESolver));
    solver->state_dim = state_dim;
    solver->config = *config;
    
    size_t num_executors = config->num_executors;
    size_t num_partitions = config->num_partitions;
    
    // Allocate executor arrays
    solver->executor_results = (double**)malloc(num_executors * sizeof(double*));
    solver->executor_loads = (size_t*)malloc(num_executors * sizeof(size_t));
    solver->executor_status = (int*)malloc(num_executors * sizeof(int));
    solver->executor_utilization = (double*)malloc(num_executors * sizeof(double));
    
    if (!solver->executor_results || !solver->executor_loads ||
        !solver->executor_status || !solver->executor_utilization) {
        spark_ode_free(solver);
        return -1;
    }
    
    // Initialize executors
    size_t partition_size = (state_dim + num_partitions - 1) / num_partitions;
    for (size_t i = 0; i < num_executors; i++) {
        solver->executor_results[i] = (double*)malloc(partition_size * sizeof(double));
        solver->executor_loads[i] = 0;
        solver->executor_status[i] = 0; // 0 = active
        solver->executor_utilization[i] = 0.0;
        
        if (!solver->executor_results[i]) {
            spark_ode_free(solver);
            return -1;
        }
    }
    
    // Initialize checkpoint storage
    if (config->enable_checkpointing) {
        size_t max_checkpoints = 10; // Maximum checkpoints to store
        solver->checkpoint_rdds = (SparkRDD**)malloc(max_checkpoints * sizeof(SparkRDD*));
        solver->num_checkpoints = 0;
        
        if (!solver->checkpoint_rdds) {
            spark_ode_free(solver);
            return -1;
        }
    }
    
    solver->cache_hit_rate = 0.0;
    solver->num_tasks = 0;
    solver->num_completed_tasks = 0;
    
    return 0;
}

void spark_ode_free(SparkODESolver* solver) {
    if (!solver) return;
    
    if (solver->executor_results) {
        for (size_t i = 0; i < solver->config.num_executors; i++) {
            if (solver->executor_results[i]) free(solver->executor_results[i]);
        }
        free(solver->executor_results);
    }
    
    if (solver->executor_loads) free(solver->executor_loads);
    if (solver->executor_status) free(solver->executor_status);
    if (solver->executor_utilization) free(solver->executor_utilization);
    
    if (solver->state_rdd) spark_free_rdd(solver->state_rdd);
    if (solver->derivative_rdd) spark_free_rdd(solver->derivative_rdd);
    if (solver->result_rdd) spark_free_rdd(solver->result_rdd);
    
    if (solver->checkpoint_rdds) {
        for (size_t i = 0; i < solver->num_checkpoints; i++) {
            if (solver->checkpoint_rdds[i]) spark_free_rdd(solver->checkpoint_rdds[i]);
        }
        free(solver->checkpoint_rdds);
    }
    
    memset(solver, 0, sizeof(SparkODESolver));
}

int spark_ode_solve(SparkODESolver* solver, ODEFunction f,
                   double t0, double t_end, const double* y0,
                   double h, void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    clock_t start_total = clock();
    clock_t start_map, start_reduce, start_shuffle;
    
    size_t state_dim = solver->state_dim;
    size_t num_partitions = solver->config.num_partitions;
    size_t num_executors = solver->config.num_executors;
    
    // ============================================================
    // CREATE RDD: Partition state data
    // Time Complexity: O(n) where n=state_dim
    // Space Complexity: O(n) for RDD storage
    // ============================================================
    if (solver->state_rdd) {
        spark_free_rdd(solver->state_rdd);
    }
    
    solver->state_rdd = spark_create_rdd(y0, state_dim, num_partitions);
    if (!solver->state_rdd) {
        return -1;
    }
    
    // Cache RDD if enabled
    if (solver->config.enable_caching) {
        spark_cache_rdd(solver->state_rdd);
    }
    
    // ============================================================
    // MAP PHASE: Transform partitions in parallel
    // Time Complexity: O(n/p) where p=partitions (parallel execution)
    // Space Complexity: O(n) for intermediate results
    // ============================================================
    start_map = clock();
    
    // Map: process each partition (in parallel in real Spark)
    size_t partition_size = (state_dim + num_partitions - 1) / num_partitions;
    size_t tasks_per_executor = (num_partitions + num_executors - 1) / num_executors;
    
    solver->num_tasks = num_partitions;
    solver->num_completed_tasks = 0;
    
    for (size_t i = 0; i < num_partitions; i++) {
        size_t executor_id = i % num_executors;
        size_t start_idx = i * partition_size;
        size_t end_idx = (start_idx + partition_size < state_dim) ?
                        start_idx + partition_size : state_dim;
        size_t local_size = end_idx - start_idx;
        
        // Map operation: compute derivative for partition
        double* partition_input = solver->state_rdd->partition_data[i];
        double* partition_output = (double*)malloc(local_size * sizeof(double));
        
        if (partition_output) {
            // Compute ODE derivative for this partition
            double* dydt = (double*)malloc(local_size * sizeof(double));
            if (dydt) {
                // Create full state for ODE function call
                double* full_state = (double*)malloc(state_dim * sizeof(double));
                if (full_state) {
                    memcpy(full_state, y0, state_dim * sizeof(double));
                    f(t0, full_state, dydt, params);
                    
                    // Extract partition derivative
                    memcpy(partition_output, &dydt[start_idx], local_size * sizeof(double));
                    free(full_state);
                }
                free(dydt);
            }
            
            // Store in executor result
            if (solver->executor_results[executor_id]) {
                memcpy(solver->executor_results[executor_id], partition_output, local_size * sizeof(double));
            }
            
            free(partition_output);
        }
        
        solver->executor_loads[executor_id]++;
        solver->num_completed_tasks++;
        
        // Update utilization
        solver->executor_utilization[executor_id] = (double)solver->executor_loads[executor_id] / tasks_per_executor;
    }
    
    solver->map_time = ((double)(clock() - start_map)) / CLOCKS_PER_SEC;
    
    // ============================================================
    // SHUFFLE PHASE: Exchange data between executors
    // Time Complexity: O(p * log(p)) for network communication
    // Space Complexity: O(n) for shuffled data
    // ============================================================
    start_shuffle = clock();
    
    // Shuffle: exchange data between partitions (simulated)
    double data_size_mb = (double)(state_dim * sizeof(double)) / (1024.0 * 1024.0);
    double network_time = data_size_mb / solver->config.network_bandwidth;
    
    // Checkpoint if enabled
    if (solver->config.enable_checkpointing) {
        double elapsed = ((double)(clock() - start_total)) / CLOCKS_PER_SEC;
        if (elapsed >= solver->config.checkpoint_interval && solver->num_checkpoints < 10) {
            // Create checkpoint (simplified)
            solver->checkpoint_rdds[solver->num_checkpoints] = spark_create_rdd(y0, state_dim, num_partitions);
            if (solver->checkpoint_rdds[solver->num_checkpoints]) {
                spark_checkpoint_rdd(solver->checkpoint_rdds[solver->num_checkpoints]);
                solver->num_checkpoints++;
            }
        }
    }
    
    solver->shuffle_time = ((double)(clock() - start_shuffle)) / CLOCKS_PER_SEC + network_time;
    
    // ============================================================
    // REDUCE PHASE: Aggregate results from executors
    // Time Complexity: O(p/e) where e=executors (parallel reduction)
    // Space Complexity: O(n) for final result
    // ============================================================
    start_reduce = clock();
    
    // Reduce: aggregate executor results
    if (!solver->result_rdd) {
        double* aggregated = (double*)malloc(state_dim * sizeof(double));
        if (aggregated) {
            memset(aggregated, 0, state_dim * sizeof(double));
            
            // Aggregate from all executors
            for (size_t i = 0; i < num_executors; i++) {
                if (solver->executor_results[i]) {
                    size_t start_idx = i * partition_size;
                    size_t end_idx = (start_idx + partition_size < state_dim) ?
                                    start_idx + partition_size : state_dim;
                    
                    for (size_t j = 0; j < end_idx - start_idx; j++) {
                        if (start_idx + j < state_dim) {
                            aggregated[start_idx + j] += solver->executor_results[i][j];
                        }
                    }
                }
            }
            
            solver->result_rdd = spark_create_rdd(aggregated, state_dim, num_partitions);
            free(aggregated);
        }
    } else {
        // Use cached result (cache hit)
        solver->cache_hit_rate = 1.0;
    }
    
    solver->reduce_time = ((double)(clock() - start_reduce)) / CLOCKS_PER_SEC;
    
    // Apply ODE step using RK3
    if (solver->result_rdd) {
        double* temp_state = (double*)malloc(state_dim * sizeof(double));
        if (temp_state) {
            // Reconstruct state from RDD
            size_t idx = 0;
            for (size_t i = 0; i < solver->result_rdd->num_partitions; i++) {
                memcpy(&temp_state[idx], solver->result_rdd->partition_data[i],
                      solver->result_rdd->partition_sizes[i] * sizeof(double));
                idx += solver->result_rdd->partition_sizes[i];
            }
            
            // Apply RK3 step
            double* t_out = (double*)malloc(sizeof(double));
            double* y_temp = (double*)malloc(state_dim * sizeof(double));
            
            if (t_out && y_temp) {
                rk3_solve(f, t0, t0 + h, temp_state, state_dim, h, params, t_out, y_temp);
                memcpy(y_out, y_temp, state_dim * sizeof(double));
                free(y_temp);
            }
            
            if (t_out) free(t_out);
            free(temp_state);
        }
    } else {
        memcpy(y_out, y0, state_dim * sizeof(double));
    }
    
    return 0;
}

int spark_pde_init(SparkPDESolver* solver, size_t spatial_dim,
                  const size_t* grid_size, const SparkConfig* config) {
    if (!solver || spatial_dim == 0 || !grid_size || !config) {
        return -1;
    }
    
    memset(solver, 0, sizeof(SparkPDESolver));
    solver->spatial_dim = spatial_dim;
    solver->config = *config;
    
    solver->grid_size = (size_t*)malloc(spatial_dim * sizeof(size_t));
    solver->spatial_partitions = (size_t*)malloc(spatial_dim * sizeof(size_t));
    
    if (!solver->grid_size || !solver->spatial_partitions) {
        spark_pde_free(solver);
        return -1;
    }
    
    size_t total_points = 1;
    for (size_t i = 0; i < spatial_dim; i++) {
        solver->grid_size[i] = grid_size[i];
        solver->spatial_partitions[i] = config->num_partitions; // Simplified
        total_points *= grid_size[i];
    }
    
    return 0;
}

void spark_pde_free(SparkPDESolver* solver) {
    if (!solver) return;
    
    if (solver->grid_size) free(solver->grid_size);
    if (solver->spatial_partitions) free(solver->spatial_partitions);
    
    if (solver->spatial_rdd) spark_free_rdd(solver->spatial_rdd);
    if (solver->spatial_derivative_rdd) spark_free_rdd(solver->spatial_derivative_rdd);
    
    memset(solver, 0, sizeof(SparkPDESolver));
}

int spark_pde_solve(SparkPDESolver* solver, PDEFunction f,
                   double t0, double t_end, const double* u0,
                   double h, void* params, double* u_out) {
    if (!solver || !f || !u0 || !u_out) {
        return -1;
    }
    
    // Similar to ODE solve but for spatial domains
    size_t total_points = 1;
    for (size_t i = 0; i < solver->spatial_dim; i++) {
        total_points *= solver->grid_size[i];
    }
    
    // Create spatial RDD
    solver->spatial_rdd = spark_create_rdd(u0, total_points, solver->config.num_partitions);
    
    if (solver->spatial_rdd) {
        // Process spatial RDD (simplified)
        memcpy(u_out, u0, total_points * sizeof(double));
    }
    
    return 0;
}

void spark_get_metrics(const SparkODESolver* solver,
                      double* map_time, double* reduce_time,
                      double* shuffle_time, double* cache_hit_rate,
                      double* total_time) {
    if (!solver) return;
    
    if (map_time) *map_time = solver->map_time;
    if (reduce_time) *reduce_time = solver->reduce_time;
    if (shuffle_time) *shuffle_time = solver->shuffle_time;
    if (cache_hit_rate) *cache_hit_rate = solver->cache_hit_rate;
    if (total_time) *total_time = solver->map_time + solver->reduce_time + solver->shuffle_time;
}

double spark_estimate_cost(const SparkODESolver* solver,
                          double* compute_hours, double* network_cost,
                          double* storage_cost) {
    if (!solver) return 0.0;
    
    double total_time = solver->map_time + solver->reduce_time + solver->shuffle_time;
    double hours = total_time / 3600.0;
    
    size_t num_executors = solver->config.num_executors;
    
    if (compute_hours) *compute_hours = hours * num_executors;
    
    // Network cost: data transfer during shuffle
    double data_size_mb = (double)(solver->state_dim * sizeof(double)) / (1024.0 * 1024.0);
    double network_transfer_cost = data_size_mb * 0.01; // $0.01 per MB
    
    if (network_cost) *network_cost = network_transfer_cost;
    
    // Storage cost: RDD caching and checkpointing
    double storage_mb = data_size_mb * (1.0 + (solver->config.enable_caching ? 1.0 : 0.0) +
                                       (solver->config.enable_checkpointing ? solver->num_checkpoints : 0.0));
    double storage_cost_val = storage_mb * 0.001; // $0.001 per MB per hour
    
    if (storage_cost) *storage_cost = storage_cost_val * hours;
    
    // Compute cost: hours * cost_per_hour * num_executors
    double compute_cost = hours * solver->config.compute_cost_per_hour * num_executors;
    
    return compute_cost + network_transfer_cost + storage_cost_val * hours;
}
