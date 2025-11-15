/*
 * Map/Reduce Framework Implementation for ODE/PDE Solving
 * Designed for commodity hardware with redundancy
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "mapreduce_solvers.h"
#include "rk3.h"
#include "euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// Default map function: compute derivative for chunk
static size_t default_map_func(const char* key, const double* value, size_t value_size,
                               double* output, size_t output_size, void* params) {
    if (!value || !output || value_size == 0 || output_size < value_size) {
        return 0;
    }
    
    // Simple map: copy and scale (in real implementation, would compute ODE derivative)
    for (size_t i = 0; i < value_size; i++) {
        output[i] = value[i] * 1.0; // Placeholder - would call ODE function
    }
    
    return value_size;
}

// Default reduce function: sum aggregation
static size_t default_reduce_func(const char* key, const double* values, size_t num_values,
                                  double* output, void* params) {
    if (!values || !output || num_values == 0) {
        return 0;
    }
    
    // Simple reduce: sum all values
    double sum = 0.0;
    for (size_t i = 0; i < num_values; i++) {
        sum += values[i];
    }
    *output = sum;
    
    return 1;
}

int mapreduce_ode_init(MapReduceODESolver* solver, size_t state_dim,
                       const MapReduceConfig* config) {
    if (!solver || state_dim == 0 || !config) {
        return -1;
    }
    
    memset(solver, 0, sizeof(MapReduceODESolver));
    solver->state_dim = state_dim;
    solver->config = *config;
    solver->map_func = default_map_func;
    solver->reduce_func = default_reduce_func;
    
    size_t num_mappers = config->num_mappers;
    size_t num_reducers = config->num_reducers;
    size_t chunk_size = (state_dim + num_mappers - 1) / num_mappers; // Ceiling division
    
    // Allocate mapper arrays
    solver->map_inputs = (double**)malloc(num_mappers * sizeof(double*));
    solver->map_outputs = (double**)malloc(num_mappers * sizeof(double*));
    solver->map_output_sizes = (size_t*)malloc(num_mappers * sizeof(size_t));
    solver->mapper_status = (int*)malloc(num_mappers * sizeof(int));
    solver->node_assignments = (size_t*)malloc(num_mappers * sizeof(size_t));
    solver->node_loads = (double*)malloc(num_mappers * sizeof(double));
    
    if (!solver->map_inputs || !solver->map_outputs || !solver->map_output_sizes ||
        !solver->mapper_status || !solver->node_assignments || !solver->node_loads) {
        mapreduce_ode_free(solver);
        return -1;
    }
    
    // Initialize mapper status (all active)
    for (size_t i = 0; i < num_mappers; i++) {
        solver->mapper_status[i] = 0; // 0 = active
        solver->node_assignments[i] = i;
        solver->node_loads[i] = 0.0;
    }
    
    // Allocate input/output for each mapper
    for (size_t i = 0; i < num_mappers; i++) {
        solver->map_inputs[i] = (double*)malloc(chunk_size * sizeof(double));
        solver->map_outputs[i] = (double*)malloc(chunk_size * sizeof(double));
        solver->map_output_sizes[i] = chunk_size;
        
        if (!solver->map_inputs[i] || !solver->map_outputs[i]) {
            mapreduce_ode_free(solver);
            return -1;
        }
    }
    
    // Allocate reducer arrays
    solver->reducer_status = (int*)malloc(num_reducers * sizeof(int));
    solver->reduce_output = (double*)malloc(state_dim * sizeof(double));
    
    if (!solver->reducer_status || !solver->reduce_output) {
        mapreduce_ode_free(solver);
        return -1;
    }
    
    for (size_t i = 0; i < num_reducers; i++) {
        solver->reducer_status[i] = 0; // 0 = active
    }
    
    // Allocate redundant outputs if redundancy enabled
    if (config->enable_redundancy) {
        size_t redundancy = config->redundancy_factor;
        solver->redundant_outputs = (double**)malloc(num_mappers * redundancy * sizeof(double*));
        if (!solver->redundant_outputs) {
            mapreduce_ode_free(solver);
            return -1;
        }
        
        for (size_t i = 0; i < num_mappers * redundancy; i++) {
            solver->redundant_outputs[i] = (double*)malloc(chunk_size * sizeof(double));
            if (!solver->redundant_outputs[i]) {
                mapreduce_ode_free(solver);
                return -1;
            }
        }
    }
    
    return 0;
}

void mapreduce_ode_free(MapReduceODESolver* solver) {
    if (!solver) return;
    
    if (solver->map_inputs) {
        for (size_t i = 0; i < solver->config.num_mappers; i++) {
            if (solver->map_inputs[i]) free(solver->map_inputs[i]);
        }
        free(solver->map_inputs);
    }
    
    if (solver->map_outputs) {
        for (size_t i = 0; i < solver->config.num_mappers; i++) {
            if (solver->map_outputs[i]) free(solver->map_outputs[i]);
        }
        free(solver->map_outputs);
    }
    
    if (solver->map_output_sizes) free(solver->map_output_sizes);
    if (solver->mapper_status) free(solver->mapper_status);
    if (solver->reducer_status) free(solver->reducer_status);
    if (solver->reduce_output) free(solver->reduce_output);
    if (solver->node_assignments) free(solver->node_assignments);
    if (solver->node_loads) free(solver->node_loads);
    
    if (solver->redundant_outputs) {
        size_t total = solver->config.num_mappers * solver->config.redundancy_factor;
        for (size_t i = 0; i < total; i++) {
            if (solver->redundant_outputs[i]) free(solver->redundant_outputs[i]);
        }
        free(solver->redundant_outputs);
    }
    
    memset(solver, 0, sizeof(MapReduceODESolver));
}

int mapreduce_ode_solve(MapReduceODESolver* solver, ODEFunction f,
                        double t0, double t_end, const double* y0,
                        double h, void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    clock_t start_total = clock();
    clock_t start_map, start_reduce, start_shuffle;
    
    size_t num_mappers = solver->config.num_mappers;
    size_t num_reducers = solver->config.num_reducers;
    size_t state_dim = solver->state_dim;
    size_t chunk_size = (state_dim + num_mappers - 1) / num_mappers;
    
    // ============================================================
    // MAP PHASE: Distribute state across mappers
    // Time Complexity: O(n/m) where n=state_dim, m=mappers
    // Space Complexity: O(n) for input distribution
    // ============================================================
    start_map = clock();
    
    // Distribute input data to mappers
    for (size_t i = 0; i < num_mappers; i++) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = (start_idx + chunk_size < state_dim) ? start_idx + chunk_size : state_dim;
        size_t local_size = end_idx - start_idx;
        
        // Copy chunk to mapper input
        memcpy(solver->map_inputs[i], &y0[start_idx], local_size * sizeof(double));
        
        // Execute map function (in parallel in real implementation)
        char key[64];
        snprintf(key, sizeof(key), "chunk_%zu", i);
        
        size_t output_size = solver->map_func(key, solver->map_inputs[i], local_size,
                                              solver->map_outputs[i],
                                              solver->map_output_sizes[i], params);
        solver->map_output_sizes[i] = output_size;
        
        // Update node load
        solver->node_loads[i] += (double)local_size / state_dim;
    }
    
    // Redundant computation for fault tolerance
    if (solver->config.enable_redundancy) {
        size_t redundancy = solver->config.redundancy_factor;
        solver->total_mappers_used = num_mappers * redundancy;
        
        for (size_t r = 0; r < redundancy; r++) {
            for (size_t i = 0; i < num_mappers; i++) {
                size_t redundant_idx = r * num_mappers + i;
                size_t start_idx = i * chunk_size;
                size_t end_idx = (start_idx + chunk_size < state_dim) ? start_idx + chunk_size : state_dim;
                size_t local_size = end_idx - start_idx;
                
                // Redundant map computation
                char key[64];
                snprintf(key, sizeof(key), "chunk_%zu_redun_%zu", i, r);
                
                solver->map_func(key, solver->map_inputs[i], local_size,
                                solver->redundant_outputs[redundant_idx],
                                chunk_size, params);
            }
        }
    } else {
        solver->total_mappers_used = num_mappers;
    }
    
    solver->map_time = ((double)(clock() - start_map)) / CLOCKS_PER_SEC;
    
    // ============================================================
    // SHUFFLE PHASE: Organize data for reducers
    // Time Complexity: O(m * log(m)) for sorting/partitioning
    // Space Complexity: O(n) for intermediate storage
    // ============================================================
    start_shuffle = clock();
    
    // Shuffle: partition mapper outputs for reducers
    // In real implementation, this involves network communication
    // For simulation, we just organize data
    size_t reducer_chunk_size = (num_mappers + num_reducers - 1) / num_reducers;
    
    // Simulate network transfer time based on bandwidth
    double data_size_mb = (double)(state_dim * sizeof(double)) / (1024.0 * 1024.0);
    double network_time = data_size_mb / solver->config.network_bandwidth;
    
    solver->shuffle_time = ((double)(clock() - start_shuffle)) / CLOCKS_PER_SEC + network_time;
    
    // ============================================================
    // REDUCE PHASE: Aggregate results from reducers
    // Time Complexity: O(m/r) where r=reducers (parallel reduction)
    // Space Complexity: O(n) for output
    // ============================================================
    start_reduce = clock();
    
    // Reduce: aggregate mapper outputs
    for (size_t i = 0; i < num_reducers; i++) {
        size_t start_mapper = i * reducer_chunk_size;
        size_t end_mapper = (start_mapper + reducer_chunk_size < num_mappers) ?
                           start_mapper + reducer_chunk_size : num_mappers;
        
        // Aggregate results from assigned mappers
        for (size_t j = start_mapper; j < end_mapper; j++) {
            size_t output_start = j * chunk_size;
            size_t output_end = (output_start + chunk_size < state_dim) ?
                               output_start + chunk_size : state_dim;
            
            // Copy to final output (in real implementation, would reduce)
            for (size_t k = 0; k < output_end - output_start; k++) {
                if (output_start + k < state_dim) {
                    solver->reduce_output[output_start + k] = solver->map_outputs[j][k];
                }
            }
        }
    }
    
    solver->total_reducers_used = num_reducers;
    solver->reduce_time = ((double)(clock() - start_reduce)) / CLOCKS_PER_SEC;
    
    // Apply ODE step using RK3 on reduced result
    // In real implementation, this would be integrated into map/reduce
    double* temp_state = (double*)malloc(state_dim * sizeof(double));
    if (temp_state) {
        memcpy(temp_state, solver->reduce_output, state_dim * sizeof(double));
        
        // Use RK3 for time stepping
        double* t_out = (double*)malloc(sizeof(double));
        double* y_temp = (double*)malloc(state_dim * sizeof(double));
        
        if (t_out && y_temp) {
            rk3_solve(f, t0, t0 + h, temp_state, state_dim, h, params, t_out, y_temp);
            memcpy(y_out, y_temp, state_dim * sizeof(double));
            free(y_temp);
        }
        
        if (t_out) free(t_out);
        free(temp_state);
    } else {
        memcpy(y_out, solver->reduce_output, state_dim * sizeof(double));
    }
    
    return 0;
}

int mapreduce_pde_init(MapReducePDESolver* solver, size_t spatial_dim,
                      const size_t* grid_size, const MapReduceConfig* config) {
    if (!solver || spatial_dim == 0 || !grid_size || !config) {
        return -1;
    }
    
    memset(solver, 0, sizeof(MapReducePDESolver));
    solver->spatial_dim = spatial_dim;
    solver->config = *config;
    solver->map_func = default_map_func;
    solver->reduce_func = default_reduce_func;
    
    solver->grid_size = (size_t*)malloc(spatial_dim * sizeof(size_t));
    solver->spatial_chunks = (size_t*)malloc(spatial_dim * sizeof(size_t));
    
    if (!solver->grid_size || !solver->spatial_chunks) {
        mapreduce_pde_free(solver);
        return -1;
    }
    
    size_t total_points = 1;
    for (size_t i = 0; i < spatial_dim; i++) {
        solver->grid_size[i] = grid_size[i];
        solver->spatial_chunks[i] = config->num_mappers; // Simplified
        total_points *= grid_size[i];
    }
    
    // Allocate spatial data chunks
    solver->spatial_data = (double***)malloc(config->num_mappers * sizeof(double**));
    solver->map_outputs = (double**)malloc(config->num_mappers * sizeof(double*));
    
    if (!solver->spatial_data || !solver->map_outputs) {
        mapreduce_pde_free(solver);
        return -1;
    }
    
    size_t chunk_points = total_points / config->num_mappers;
    for (size_t i = 0; i < config->num_mappers; i++) {
        solver->spatial_data[i] = (double**)malloc(chunk_points * sizeof(double*));
        solver->map_outputs[i] = (double*)malloc(chunk_points * sizeof(double));
        
        if (!solver->spatial_data[i] || !solver->map_outputs[i]) {
            mapreduce_pde_free(solver);
            return -1;
        }
    }
    
    solver->reduce_output = (double*)malloc(total_points * sizeof(double));
    if (!solver->reduce_output) {
        mapreduce_pde_free(solver);
        return -1;
    }
    
    return 0;
}

void mapreduce_pde_free(MapReducePDESolver* solver) {
    if (!solver) return;
    
    if (solver->grid_size) free(solver->grid_size);
    if (solver->spatial_chunks) free(solver->spatial_chunks);
    if (solver->reduce_output) free(solver->reduce_output);
    
    if (solver->spatial_data) {
        for (size_t i = 0; i < solver->config.num_mappers; i++) {
            if (solver->spatial_data[i]) free(solver->spatial_data[i]);
        }
        free(solver->spatial_data);
    }
    
    if (solver->map_outputs) {
        for (size_t i = 0; i < solver->config.num_mappers; i++) {
            if (solver->map_outputs[i]) free(solver->map_outputs[i]);
        }
        free(solver->map_outputs);
    }
    
    memset(solver, 0, sizeof(MapReducePDESolver));
}

int mapreduce_pde_solve(MapReducePDESolver* solver, PDEFunction f,
                        double t0, double t_end, const double* u0,
                        double h, void* params, double* u_out) {
    if (!solver || !f || !u0 || !u_out) {
        return -1;
    }
    
    // Similar to ODE solve but for spatial domains
    // Map phase: process spatial chunks
    // Reduce phase: aggregate spatial results
    
    size_t total_points = 1;
    for (size_t i = 0; i < solver->spatial_dim; i++) {
        total_points *= solver->grid_size[i];
    }
    
    clock_t start_map = clock();
    
    // Map: process each spatial chunk
    size_t chunk_points = total_points / solver->config.num_mappers;
    for (size_t i = 0; i < solver->config.num_mappers; i++) {
        size_t start_idx = i * chunk_points;
        size_t end_idx = (i + 1) * chunk_points;
        if (end_idx > total_points) end_idx = total_points;
        
        // Process spatial chunk (simplified)
        for (size_t j = start_idx; j < end_idx; j++) {
            solver->map_outputs[i][j - start_idx] = u0[j];
        }
    }
    
    solver->map_time = ((double)(clock() - start_map)) / CLOCKS_PER_SEC;
    
    // Shuffle and reduce (simplified)
    clock_t start_reduce = clock();
    memcpy(solver->reduce_output, u0, total_points * sizeof(double));
    solver->reduce_time = ((double)(clock() - start_reduce)) / CLOCKS_PER_SEC;
    
    memcpy(u_out, solver->reduce_output, total_points * sizeof(double));
    
    return 0;
}

void mapreduce_get_metrics(const MapReduceODESolver* solver,
                          double* map_time, double* reduce_time,
                          double* shuffle_time, double* total_time) {
    if (!solver) return;
    
    if (map_time) *map_time = solver->map_time;
    if (reduce_time) *reduce_time = solver->reduce_time;
    if (shuffle_time) *shuffle_time = solver->shuffle_time;
    if (total_time) *total_time = solver->map_time + solver->reduce_time + solver->shuffle_time;
}

double mapreduce_estimate_cost(const MapReduceODESolver* solver,
                               double* compute_hours, double* network_cost) {
    if (!solver) return 0.0;
    
    double total_time = solver->map_time + solver->reduce_time + solver->shuffle_time;
    double hours = total_time / 3600.0;
    
    if (compute_hours) *compute_hours = hours * solver->total_mappers_used;
    
    // Network cost: data transfer during shuffle
    double data_size_mb = (double)(solver->state_dim * sizeof(double)) / (1024.0 * 1024.0);
    double network_transfer_cost = data_size_mb * 0.01; // $0.01 per MB (example)
    
    if (network_cost) *network_cost = network_transfer_cost;
    
    // Compute cost: hours * cost_per_hour * num_nodes
    double compute_cost = hours * solver->config.compute_cost_per_hour * solver->total_mappers_used;
    
    return compute_cost + network_transfer_cost;
}
