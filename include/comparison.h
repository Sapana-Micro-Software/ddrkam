/*
 * Method Comparison Framework
 * RK3 vs DDRK3 vs AM vs DDAM
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef COMPARISON_H
#define COMPARISON_H

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
 * Comparison results structure
 */
typedef struct {
    // Standard methods
    double euler_time;
    double ddeuler_time;
    double rk3_time;
    double ddrk3_time;
    double am_time;
    double ddam_time;
    
    // Parallel methods
    double parallel_rk3_time;
    double parallel_am_time;
    double parallel_euler_time;
    double stacked_rk3_time;
    double stacked_am_time;
    double stacked_euler_time;
    
    // Errors
    double euler_error;
    double ddeuler_error;
    double rk3_error;
    double ddrk3_error;
    double am_error;
    double ddam_error;
    double parallel_rk3_error;
    double parallel_am_error;
    double parallel_euler_error;
    double stacked_rk3_error;
    double stacked_am_error;
    double stacked_euler_error;
    
    // Accuracies
    double euler_accuracy;
    double ddeuler_accuracy;
    double rk3_accuracy;
    double ddrk3_accuracy;
    double am_accuracy;
    double ddam_accuracy;
    double parallel_rk3_accuracy;
    double parallel_am_accuracy;
    double parallel_euler_accuracy;
    double stacked_rk3_accuracy;
    double stacked_am_accuracy;
    double stacked_euler_accuracy;
    
    // Steps
    size_t euler_steps;
    size_t ddeuler_steps;
    size_t rk3_steps;
    size_t ddrk3_steps;
    size_t am_steps;
    size_t ddam_steps;
    size_t parallel_rk3_steps;
    size_t parallel_am_steps;
    size_t parallel_euler_steps;
    size_t stacked_rk3_steps;
    size_t stacked_am_steps;
    size_t stacked_euler_steps;
    
    // Real-time methods
    double realtime_rk3_time;
    double realtime_am_time;
    double realtime_euler_time;
    double realtime_rk3_error;
    double realtime_am_error;
    double realtime_euler_error;
    double realtime_rk3_accuracy;
    double realtime_am_accuracy;
    double realtime_euler_accuracy;
    
    // Online methods
    double online_rk3_time;
    double online_am_time;
    double online_euler_time;
    double online_rk3_error;
    double online_am_error;
    double online_euler_error;
    double online_rk3_accuracy;
    double online_am_accuracy;
    double online_euler_accuracy;
    
    // Dynamic methods
    double dynamic_rk3_time;
    double dynamic_am_time;
    double dynamic_euler_time;
    double dynamic_rk3_error;
    double dynamic_am_error;
    double dynamic_euler_error;
    double dynamic_rk3_accuracy;
    double dynamic_am_accuracy;
    double dynamic_euler_accuracy;
    
    // Nonlinear programming solvers
    double nonlinear_ode_time;
    double nonlinear_pde_time;
    
    // Karmarkar's Algorithm
    double karmarkar_time;
    double karmarkar_error;
    double karmarkar_accuracy;
    size_t karmarkar_steps;
    size_t karmarkar_iterations;
    
    // Map/Reduce Framework
    double mapreduce_time;
    double mapreduce_error;
    double mapreduce_accuracy;
    size_t mapreduce_steps;
    double mapreduce_map_time;
    double mapreduce_reduce_time;
    double mapreduce_shuffle_time;
    double mapreduce_cost;
    
    // Spark Framework
    double spark_time;
    double spark_error;
    double spark_accuracy;
    size_t spark_steps;
    double spark_map_time;
    double spark_reduce_time;
    double spark_shuffle_time;
    double spark_cache_hit_rate;
    double spark_cost;
    double nonlinear_ode_error;
    double nonlinear_pde_error;
    double nonlinear_ode_accuracy;
    double nonlinear_pde_accuracy;
    
    // Additional distributed/data-driven/online/real-time solvers
    double distributed_datadriven_time;
    double online_datadriven_time;
    double realtime_datadriven_time;
    double distributed_online_time;
    double distributed_realtime_time;
    
    // Parallel performance metrics
    double speedup_rk3;      // Speedup factor for parallel RK3
    double speedup_am;        // Speedup factor for parallel AM
    double speedup_euler;    // Speedup factor for parallel Euler
    size_t num_workers;       // Number of parallel workers used
    
    // Non-orthodox architectures
    double microgasjet_time;
    double microgasjet_error;
    double microgasjet_accuracy;
    size_t microgasjet_steps;
    double microgasjet_flow_energy;
    
    double dataflow_time;
    double dataflow_error;
    double dataflow_accuracy;
    size_t dataflow_steps;
    size_t dataflow_tokens;
    double dataflow_token_matching_time;
    
    double ace_time;
    double ace_error;
    double ace_accuracy;
    size_t ace_steps;
    size_t ace_instructions;
    double ace_memory_time;
    
    double systolic_time;
    double systolic_error;
    double systolic_accuracy;
    size_t systolic_steps;
    double systolic_communication_time;
    
    double tpu_time;
    double tpu_error;
    double tpu_accuracy;
    size_t tpu_steps;
    size_t tpu_matrix_ops;
    double tpu_bandwidth_utilization;
    
    double gpu_cuda_time;
    double gpu_cuda_error;
    double gpu_cuda_accuracy;
    size_t gpu_cuda_steps;
    size_t gpu_cuda_kernel_launches;
    double gpu_cuda_memory_transfer_time;
    
    double gpu_metal_time;
    double gpu_metal_error;
    double gpu_metal_accuracy;
    size_t gpu_metal_steps;
    
    double gpu_vulkan_time;
    double gpu_vulkan_error;
    double gpu_vulkan_accuracy;
    size_t gpu_vulkan_steps;
    
    double gpu_amd_time;
    double gpu_amd_error;
    double gpu_amd_accuracy;
    size_t gpu_amd_steps;
    
    // Additional non-orthodox architectures
    double massively_threaded_time;
    double massively_threaded_error;
    double massively_threaded_accuracy;
    size_t massively_threaded_steps;
    size_t massively_threaded_nodes_expanded;
    
    double starr_time;
    double starr_error;
    double starr_accuracy;
    size_t starr_steps;
    size_t starr_semantic_hits;
    size_t starr_associative_hits;
    
    double truenorth_time;
    double truenorth_error;
    double truenorth_accuracy;
    size_t truenorth_steps;
    size_t truenorth_spikes;
    double truenorth_energy;
    
    double loihi_time;
    double loihi_error;
    double loihi_accuracy;
    size_t loihi_steps;
    size_t loihi_spikes;
    
    double brainchips_time;
    double brainchips_error;
    double brainchips_accuracy;
    size_t brainchips_steps;
    size_t brainchips_events;
    
    double racetrack_time;
    double racetrack_error;
    double racetrack_accuracy;
    size_t racetrack_steps;
    size_t racetrack_domain_movements;
    
    double pcm_time;
    double pcm_error;
    double pcm_accuracy;
    size_t pcm_steps;
    size_t pcm_phase_transitions;
    
    double lyric_time;
    double lyric_error;
    double lyric_accuracy;
    size_t lyric_steps;
    size_t lyric_samples;
    
    double hw_bayesian_time;
    double hw_bayesian_error;
    double hw_bayesian_accuracy;
    size_t hw_bayesian_steps;
    size_t hw_bayesian_inference_ops;
    
    double semantic_lexo_bs_time;
    double semantic_lexo_bs_error;
    double semantic_lexo_bs_accuracy;
    size_t semantic_lexo_bs_steps;
    size_t semantic_lexo_bs_nodes_searched;
    
    double kernelized_sps_bs_time;
    double kernelized_sps_bs_error;
    double kernelized_sps_bs_accuracy;
    size_t kernelized_sps_bs_steps;
    size_t kernelized_sps_bs_kernel_evals;
    
    double spiralizer_chord_time;
    double spiralizer_chord_error;
    double spiralizer_chord_accuracy;
    size_t spiralizer_chord_steps;
    size_t spiralizer_chord_collisions;
    size_t spiralizer_chord_spiral_steps;
    
    double lattice_waterfront_time;
    double lattice_waterfront_error;
    double lattice_waterfront_accuracy;
    size_t lattice_waterfront_steps;
    size_t lattice_waterfront_routing_ops;
    size_t lattice_waterfront_waterfront_ops;
    
    double multiple_search_tree_time;
    double multiple_search_tree_error;
    double multiple_search_tree_accuracy;
    size_t multiple_search_tree_steps;
    size_t multiple_search_tree_nodes_expanded;
    size_t multiple_search_tree_nodes_generated;
} ComparisonResults;

/**
 * Run comprehensive comparison of all methods
 * 
 * @param f: ODE function
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state
 * @param n: System dimension
 * @param h: Step size
 * @param params: ODE parameters
 * @param exact_solution: Exact solution at t_end (for error calculation)
 * @param results: Output comparison results
 * @return: 0 on success, -1 on failure
 */
int compare_methods(ODEFunction f, double t0, double t_end, const double* y0,
                   size_t n, double h, void* params, const double* exact_solution,
                   ComparisonResults* results);

/**
 * Print comparison results
 */
void print_comparison_results(const ComparisonResults* results);

/**
 * Export comparison results to CSV
 * 
 * @param filename: Output CSV filename
 * @param results: Comparison results
 * @return: 0 on success, -1 on failure
 */
int export_comparison_csv(const char* filename, const ComparisonResults* results);

#ifdef __cplusplus
}
#endif

#endif /* COMPARISON_H */
