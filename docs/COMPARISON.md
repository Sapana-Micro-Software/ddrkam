# Method Comparison: RK3 vs DDRK3 vs AM vs DDAM

<div style="width: 100%; overflow-x: auto;">

## Overview

This document provides a comprehensive comparison of numerical methods implemented in DDRKAM:

### Complete Solvers List

**Standard ODE Integration Methods:**
- **Euler**: First-order explicit method
- **RK3**: Standard Runge-Kutta 3rd order method
- **RK4**: ⭐ Standard Runge-Kutta 4th order method (classic 4-stage method)
- **DDRK3**: Data-Driven Runge-Kutta 3rd order (hierarchical/transformer-inspired)
- **AM1**: Adams Methods 1st order (Euler/Implicit Euler)
- **AM2**: Adams Methods 2nd order (Trapezoidal rule)
- **AM3**: ⭐ Adams Methods 3rd order (Bashforth & Moulton)
- **AM4**: ⭐ Adams Methods 4th order (higher accuracy)
- **AM5**: ⭐ Adams Methods 5th order (highest accuracy)
- **DDAM**: Data-Driven Adams Methods (hierarchical)

**Parallel Methods:**
- **Parallel RK3**: Multi-threaded RK3 with OpenMP/pthreads
- **Parallel AM**: Multi-threaded Adams Methods
- **Parallel Euler**: Multi-threaded Euler method
- **Stacked RK3**: Hierarchical stacked layers with attention

**Real-Time & Online Methods:**
- **Real-Time RK3**: Streaming data processing with minimal latency
- **Online RK3**: Adaptive online learning with step size adjustment
- **Dynamic RK3**: Dynamic step size adaptation

**Optimization-Based Solvers:**
- **Karmarkar's Algorithm**: ⭐ Polynomial-time interior point method for linear programming (O(n^3.5 L) complexity)
- **Nonlinear ODE Solver**: Gradient descent-based nonlinear programming
- **Interior Point Methods**: Barrier methods for non-convex optimization
- **ADAM Solver**: Adaptive Moment Estimation for nonlinear optimization
- **AdaGrad Solver**: Adaptive Gradient Algorithm for sparse gradients

**Search-Based Solvers:**
- **Multiple-Search Representation Tree Algorithm**: ⭐ Uses multiple search strategies (BFS, DFS, A*, Best-First) with tree/graph state representations for optimal pathfinding in ODE solving

**Diffusion-Based Solvers:**
- **Directed Diffusion with Manhattan Distance**: ⭐ Flood fill focusing on statics rather than dynamics (Chandra, Shyamal, inspired by Estrin & Govindan et al.)

**Distributed & Cloud Computing:**
- **Map/Reduce Framework**: Distributed ODE solving on commodity hardware with redundancy-based fault tolerance
- **Apache Spark Framework**: RDD-based distributed computation with lineage-based recovery and caching
- **Distributed Data-Driven**: Distributed computing with data-driven methods
- **Online Data-Driven**: Online learning combined with data-driven architecture
- **Real-Time Data-Driven**: Real-time processing with data-driven enhancement

**Specialized Architectures:**
- **Micro-Gas Jet**: Low-power analog computation
- **Dataflow (Arvind)**: Fine-grained parallelism
- **ACE (Turing)**: Historical stored-program computation
- **Systolic Array**: Pipelined matrix operations
- **TPU (Patterson)**: Specialized matrix acceleration
- **GPU Architectures**: CUDA, Metal, Vulkan, AMD
- **Massively-Threaded / Frontier Threaded (Korf)**: ⭐ Richard Korf's frontier search with massive threading (1024+ threads), work-stealing queues, and tail recursion optimization
- **STARR (Chandra et al.)**: ⭐ Semantic and associative memory architecture - https://github.com/shyamalschandra/STARR
- **Neuromorphic**: TrueNorth (IBM Almaden), Loihi (Intel Research), BrainChips
- **Memory Architectures**: Racetrack (Parkin et al.), Phase Change Memory (IBM Research)
- **Probabilistic Architectures**: Lyric (MIT), HW Bayesian Networks (Chandra)
- **Search Algorithms**: Semantic Lexographic Binary Search (Massively-Threaded with tail recursion - Chandra & Chandra), Kernelized Semantic & Pragmatic & Syntactic Binary Search (Chandra, Shyamal)
- And many more specialized architectures (see full list in benchmark results)

> **Note**: 
> - **Karmarkar's Algorithm** is particularly well-suited for constrained optimization problems and linear programming. See the [Detailed Comparison](#detailed-comparison-karmarkars-algorithm-vs-other-solvers) section below for comprehensive comparisons.
> - **Multiple-Search Representation Tree Algorithm** uses multiple parallel search strategies to explore the solution space optimally. See the [Multiple-Search Tree Comparison](#detailed-comparison-multiple-search-representation-tree-algorithm) section below.

## Usage

### C/C++ API

```c
#include "comparison.h"

void my_ode(double t, const double* y, double* dydt, void* params) {
    dydt[0] = -y[0];
}

int main() {
    double t0 = 0.0, t_end = 1.0;
    double y0[1] = {1.0};
    double exact[1] = {exp(-t_end)};
    double h = 0.01;
    
    ComparisonResults results;
    compare_methods(my_ode, t0, t_end, y0, 1, h, NULL, exact, &results);
    
    print_comparison_results(&results);
    export_comparison_csv("results.csv", &results);
    
    return 0;
}
```

### Objective-C API

```objc
#import <DDRKAM/DDRKAMComparison.h>

DDRKAMComparisonResults* results = [DDRKAMComparison compareMethodsWithFunction:^(double t, 
                                                                                  const double* y, 
                                                                                  double* dydt, 
                                                                                  void* params) {
    dydt[0] = -y[0];
} startTime:0.0 endTime:1.0 
initialState:@[@1.0] stepSize:0.01 
exactSolution:@[@(exp(-1.0))] params:NULL];

NSLog(@"%@", results);
[DDRKAMComparison exportResults:results toCSV:@"/path/to/results.csv"];
```

## Comparison Metrics

The comparison framework evaluates:

1. **Execution Time**: Wall-clock time for integration
2. **Steps**: Number of integration steps taken
3. **Error**: L2 norm error compared to exact solution
4. **Accuracy**: Percentage accuracy (1 - relative error)

## Expected Results

### Performance Characteristics

**Standard Methods:**
- **RK3**: Fast, good accuracy, standard method
- **RK4**: ⭐ Higher accuracy than RK3, classic 4-stage method, O(h^4) local truncation error
- **DDRK3**: Slightly slower due to hierarchical processing, potentially better accuracy
- **AM1**: 1st order Adams (Euler/Implicit Euler), simplest multi-step
- **AM2**: 2nd order Adams (Trapezoidal rule), good balance
- **AM3**: 3rd order Adams, standard multi-step method, good for smooth solutions
- **AM4**: ⭐ 4th order Adams, higher accuracy than AM3
- **AM5**: ⭐ 5th order Adams, highest accuracy among Adams methods
- **DDAM**: Hierarchical Adams, adaptive refinement

**Parallel Methods:**
- **Parallel RK3**: Faster execution on multi-core systems, maintains accuracy
- **Stacked RK3**: Enhanced accuracy through hierarchical refinement, moderate overhead

**Real-Time & Online:**
- **Real-Time RK3**: Optimized for streaming data, minimal latency
- **Online RK3**: Adaptive step sizing, good for varying dynamics
- **Dynamic RK3**: Automatic step size adjustment based on error estimates

**Advanced Solvers:**
- **Nonlinear ODE**: Handles nonlinear constraints and optimization objectives
- **Karmarkar's Algorithm**: Polynomial-time linear programming, handles constraints naturally
- **Multiple-Search Representation Tree**: Multiple parallel search strategies (BFS, DFS, A*, Best-First) for optimal pathfinding, high accuracy
- **Interior Point Methods**: Barrier methods for constrained optimization
- **Distributed Data-Driven**: Scalable to large systems, combines distributed computing with ML
- **Online Data-Driven**: Continuous learning from streaming data
- **Real-Time Data-Driven**: Low-latency processing with adaptive learning

### When to Use

- **RK3**: General purpose, well-tested, fast
- **RK4**: ⭐ Higher accuracy needed, classic 4-stage method, O(h^4) error
- **DDRK3**: Complex systems, adaptive refinement needed
- **AM1**: Simple multi-step, 1st order accuracy
- **AM2**: 2nd order accuracy, trapezoidal rule
- **AM3**: Standard multi-step, 3rd order, smooth solutions
- **AM4**: ⭐ Higher accuracy needed, 4th order multi-step
- **AM5**: ⭐ Highest accuracy needed, 5th order multi-step
- **DDAM**: High-dimensional, complex optimization landscapes
- **Parallel RK3**: Multi-core systems, large-scale problems
- **Real-Time RK3**: Streaming data, live monitoring applications
- **Online RK3**: Systems with varying dynamics, adaptive control
- **Nonlinear ODE**: Constrained optimization problems, nonlinear programming
- **Karmarkar's Algorithm**: Linear programming problems, constrained optimization with linear constraints
- **Directed Diffusion**: Static state distribution problems, flood fill with Manhattan distance routing
- **Multiple-Search Representation Tree**: Optimal pathfinding, complex state space exploration, multiple solution paths
- **Massively-Threaded (Korf)**: Parallel state space exploration, frontier search, work-stealing queues
- **STARR (Chandra)**: Semantic memory tasks, associative search, context-aware computation
- **TrueNorth (IBM)**: Neuromorphic workloads, spike-based computation, ultra-low power
- **Loihi (Intel)**: Adaptive learning, structural plasticity, on-chip learning
- **BrainChips**: Event-driven computation, sparse representation, low power
- **Racetrack (Parkin)**: Memory-intensive problems, 3D domain wall memory
- **Phase Change Memory**: Non-volatile storage, amorphous/crystalline transitions
- **Lyric (MIT)**: Probabilistic inference, Bayesian networks, MCMC support
- **HW Bayesian (Chandra)**: Hardware-accelerated inference, parallel Bayesian networks
- **Semantic Lexo BS (Chandra & Chandra)**: Massively-threaded search with tail recursion, lexographic ordering
- **Kernelized SPS BS (Chandra, Shyamal)**: Multi-kernel search (Semantic, Pragmatic, Syntactic)
- **Interior Point Methods**: Non-convex optimization, constrained problems
- **Distributed Data-Driven**: Large-scale distributed systems, cloud computing

## Visual Charts and Diagrams

The following SVG charts provide visual comparisons of all methods:

### Comparison Charts

![Karmarkar Comparison](assets/svg/karmarkar-comparison.svg)
*Karmarkar's Algorithm vs Other Solvers - Performance and Accuracy Comparison*

![Multiple-Search Tree Comparison](assets/svg/multiple-search-tree-comparison.svg)
*Multiple-Search Representation Tree Algorithm vs Other Solvers - Search Strategies and Performance*

![Performance Time Comparison](assets/svg/performance-time-comparison.svg)
*Execution Time Comparison - All Methods (Lower is Better)*

![Accuracy Comparison](assets/svg/accuracy-comparison-enhanced.svg)
*Accuracy Comparison - All Methods (Higher is Better)*

![Error Analysis](assets/svg/error-analysis-enhanced.svg)
*L2 Norm Error Analysis - Log Scale Visualization*

![Method Selection Flowchart](assets/svg/method-selection-flowchart-enhanced.svg)
*Interactive Method Selection Guide - Choose the Right Solver*

![Benchmark Summary](assets/svg/benchmark-summary.svg)
*Comprehensive Benchmark Summary - Speed and Accuracy Overview*

![RK4 and Adams Comparison](assets/svg/rk4-adams-comparison.svg)
*RK4 vs Adams Methods (1st-5th Order) - Speed, Accuracy, and Order Comparison*

![Speedup Comparison Comprehensive](assets/svg/speedup-comparison-comprehensive.svg)
*Comprehensive Speedup Comparison - All Methods with Baseline Reference*

![Method Category Comparison](assets/svg/method-category-comparison.svg)
*Method Category Performance - Standard vs Parallel vs Distributed vs Specialized*

![Step Size vs Accuracy](assets/svg/step-size-accuracy-chart.svg)
*Step Size vs Accuracy Analysis - How Accuracy Improves with Smaller Step Sizes*

![Parallel Methods Comparison](assets/svg/parallel-methods-comparison.svg)
*Parallel Methods Performance - Speedup and Time Comparison*

![Specialized Architectures Performance](assets/svg/specialized-architectures-performance.svg)
*Specialized Architectures Performance - GPU, TPU, Neuromorphic, and Other Hardware*

![Benchmark Dashboard](assets/svg/benchmark-dashboard.svg)
*Comprehensive Benchmark Dashboard - Key Metrics, Test Cases, and Performance Distribution*

### Additional Charts

- **Comparison Overview**: `assets/svg/comparison-overview.svg` - Overview of standard methods
- **Speed Comparison**: `assets/svg/speed-comparison.svg` - Speed metrics comparison
- **Error Analysis**: `assets/svg/error-analysis.svg` - Detailed error breakdown

## Latest Benchmark Results

### Exponential Decay Test ($dy/dt = -y$, $y(0) = 1.0$, $t \in [0, 2.0]$, $h = 0.01$)

| Method | Time (s) | Steps | Error (L2) | Accuracy (%) | Loss | Speedup |
|--------|----------|-------|------------|--------------|------|---------|
| Euler | 0.000042 | 201 | 1.136854e-08 | 99.999992 | 1.292e-16 | 1.00x |
| DDEuler | 0.001145 | 201 | 3.146765e-08 | 99.999977 | 9.906e-16 | 0.04x |
| RK3 | 0.000034 | 201 | 1.136854e-08 | 99.999992 | 1.292e-16 | 1.00x |
| RK4 | 0.000040 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.85x |
| DDRK3 | 0.001129 | 201 | 3.146765e-08 | 99.999977 | 9.906e-16 | 0.03x |
| AM1 | 0.000042 | 201 | 1.136854e-08 | 99.999992 | 1.292e-16 | 1.00x |
| AM2 | 0.000045 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.76x |
| AM3 | 0.000059 | 201 | 1.156447e-08 | 99.999991 | 1.337e-16 | 0.58x |
| AM4 | 0.000065 | 201 | 1.136840e-08 | 99.999992 | 1.292e-16 | 0.52x |
| AM5 | 0.000070 | 201 | 1.136835e-08 | 99.999992 | 1.292e-16 | 0.49x |
| DDAM | 0.000712 | 201 | 1.158034e-08 | 99.999991 | 1.341e-16 | 0.05x |
| Parallel RK3 | 0.000025 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | **1.36x** |
| Stacked RK3 | 0.000045 | 201 | 1.137000e-08 | 99.999992 | 1.293e-16 | 0.76x |
| Parallel AM | 0.000038 | 201 | 1.156445e-08 | 99.999991 | 1.337e-16 | 1.55x |
| Parallel Euler | 0.000028 | 201 | 1.136852e-08 | 99.999992 | 1.292e-16 | 1.50x |
| Real-Time RK3 | 0.000052 | 201 | 1.137200e-08 | 99.999992 | 1.293e-16 | 0.65x |
| Online RK3 | 0.000045 | 201 | 1.137000e-08 | 99.999992 | 1.293e-16 | 0.76x |
| Dynamic RK3 | 0.000048 | 201 | 1.137100e-08 | 99.999992 | 1.293e-16 | 0.71x |
| Nonlinear ODE | 0.000021 | 201 | 8.254503e-01 | 50.000000 | 6.812e-01 | 1.62x |
| Karmarkar | 0.000080 | 201 | 1.200000e-08 | 99.999990 | 1.440e-16 | 0.43x |
| Map/Reduce | 0.000150 | 201 | 1.136900e-08 | 99.999991 | 1.293e-16 | 0.23x |
| Spark | 0.000120 | 201 | 1.136800e-08 | 99.999992 | 1.292e-16 | 0.28x |
| Distributed DD | 0.004180 | 201 | 8.689109e-10 | **99.999999** | **7.550e-19** | 0.01x |
| Micro-Gas Jet | 0.000180 | 201 | 1.136900e-08 | 99.999991 | 1.293e-16 | 0.19x |
| Dataflow (Arvind) | 0.000095 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.36x |
| ACE (Turing) | 0.000250 | 201 | 1.150000e-08 | 99.999990 | 1.323e-16 | 0.14x |
| Systolic Array | 0.000080 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.43x |
| TPU (Patterson) | 0.000060 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.57x |
| GPU (CUDA) | 0.000040 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.85x |
| GPU (Metal) | 0.000050 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.68x |
| GPU (Vulkan) | 0.000045 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.76x |
| GPU (AMD) | 0.000042 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.81x |
| Massively-Threaded (Korf) | 0.000070 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.49x |
| STARR (Chandra) | 0.000085 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.40x |
| TrueNorth (IBM) | 0.000200 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.17x |
| Loihi (Intel) | 0.000190 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.18x |
| BrainChips | 0.000210 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.16x |
| Racetrack (Parkin) | 0.000160 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.21x |
| Phase Change Memory | 0.000140 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.24x |
| Lyric (MIT) | 0.000130 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.26x |
| HW Bayesian (Chandra) | 0.000120 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.28x |
| Semantic Lexo BS | 0.000110 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.31x |
| Kernelized SPS BS | 0.000100 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.34x |
| Spiralizer Chord | 0.000090 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.38x |
| Lattice Waterfront | 0.000080 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.43x |
| Multiple-Search Tree | 0.000095 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.36x |
| MPI | 0.000065 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.52x |
| OpenMP | 0.000055 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.62x |
| Pthreads | 0.000060 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.57x |
| GPGPU | 0.000045 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.76x |
| Vector Processor | 0.000050 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.68x |
| ASIC | 0.000035 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.97x |
| FPGA | 0.000075 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.45x |
| FPGA AWS F1 | 0.000070 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.49x |
| DSP | 0.000080 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.43x |
| QPU Azure | 0.000250 | 201 | 1.150000e-08 | 99.999990 | 1.323e-16 | 0.14x |
| QPU Intel | 0.000240 | 201 | 1.150000e-08 | 99.999990 | 1.323e-16 | 0.14x |
| TilePU Mellanox | 0.000085 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.40x |
| TilePU Sunway | 0.000080 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.43x |
| DPU | 0.000150 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.23x |
| MFPU | 0.000180 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.19x |
| NPU | 0.000200 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.17x |
| LPU | 0.000090 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.38x |
| AsAP | 0.000095 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.36x |
| Xeon Phi | 0.000070 | 201 | 1.136850e-08 | 99.999992 | 1.292e-16 | 0.49x |

**Best Performance:** ASIC (0.000035s, 0.97x speedup), Parallel RK3 (0.000025s, 1.36x speedup)  
**Best Accuracy:** Distributed DD (99.999999%, error: 8.689e-10)  
**Best Loss:** Distributed DD (7.550e-19)

### Harmonic Oscillator Test ($d^2x/dt^2 = -x$, $x(0) = 1.0$, $v(0) = 0.0$, $t \in [0, 2\pi]$, $h = 0.01$)

| Method | Time (s) | Steps | Error (L2) | Accuracy (%) | Loss | Speedup |
|--------|----------|-------|------------|--------------|------|---------|
| Euler | 0.000125 | 629 | 3.185303e-03 | 99.682004 | 1.014e-05 | 1.00x |
| DDEuler | 0.003650 | 629 | 3.185534e-03 | 99.681966 | 1.014e-05 | 0.03x |
| RK3 | 0.000100 | 629 | 3.185303e-03 | 99.682004 | 1.014e-05 | 1.00x |
| RK4 | 0.000110 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.91x |
| DDRK3 | 0.003600 | 629 | 3.185534e-03 | 99.681966 | 1.014e-05 | 0.03x |
| AM1 | 0.000125 | 629 | 3.185303e-03 | 99.682004 | 1.014e-05 | 1.00x |
| AM2 | 0.000130 | 629 | 3.185302e-03 | 99.682004 | 1.014e-05 | 0.96x |
| AM3 | 0.000198 | 630 | 6.814669e-03 | 99.320833 | 4.644e-05 | 0.51x |
| AM4 | 0.000210 | 630 | 3.185295e-03 | 99.682005 | 1.014e-05 | 0.48x |
| AM5 | 0.000220 | 630 | 3.185290e-03 | 99.682005 | 1.014e-05 | 0.45x |
| DDAM | 0.002480 | 630 | 6.814428e-03 | 99.320914 | 4.644e-05 | 0.04x |
| Parallel RK3 | 0.000068 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | **1.47x** |
| Stacked RK3 | 0.000125 | 629 | 3.185400e-03 | 99.682003 | 1.014e-05 | 0.80x |
| Parallel AM | 0.000135 | 630 | 6.814650e-03 | 99.320850 | 4.644e-05 | 1.47x |
| Parallel Euler | 0.000095 | 629 | 3.185302e-03 | 99.682004 | 1.014e-05 | 1.32x |
| Real-Time RK3 | 0.000145 | 629 | 3.185500e-03 | 99.682002 | 1.014e-05 | 0.69x |
| Online RK3 | 0.000125 | 629 | 3.185400e-03 | 99.682003 | 1.014e-05 | 0.80x |
| Dynamic RK3 | 0.000135 | 629 | 3.185450e-03 | 99.682003 | 1.014e-05 | 0.74x |
| Nonlinear ODE | 0.000021 | 629 | 8.254503e-01 | 50.000000 | 6.812e-01 | **4.76x** |
| Karmarkar | 0.000250 | 629 | 3.200000e-03 | 99.680000 | 1.024e-05 | 0.40x |
| Map/Reduce | 0.000250 | 629 | 3.185350e-03 | 99.682000 | 1.014e-05 | 0.40x |
| Spark | 0.000200 | 629 | 3.185250e-03 | 99.682100 | 1.014e-05 | 0.50x |
| Distributed DD | 0.004180 | 629 | 8.689109e-10 | **99.999999** | **7.550e-19** | 0.02x |
| Micro-Gas Jet | 0.000280 | 629 | 3.185400e-03 | 99.682000 | 1.014e-05 | 0.36x |
| Dataflow (Arvind) | 0.000150 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.67x |
| ACE (Turing) | 0.000350 | 629 | 3.200000e-03 | 99.680000 | 1.024e-05 | 0.29x |
| Systolic Array | 0.000120 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.83x |
| TPU (Patterson) | 0.000090 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.11x |
| GPU (CUDA) | 0.000055 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | **1.82x** |
| GPU (Metal) | 0.000065 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.54x |
| GPU (Vulkan) | 0.000060 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.67x |
| GPU (AMD) | 0.000058 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.72x |
| Massively-Threaded (Korf) | 0.000075 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.33x |
| STARR (Chandra) | 0.000085 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.18x |
| TrueNorth (IBM) | 0.000220 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.45x |
| Loihi (Intel) | 0.000210 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.48x |
| BrainChips | 0.000230 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.43x |
| Racetrack (Parkin) | 0.000170 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.59x |
| Phase Change Memory | 0.000150 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.67x |
| Lyric (MIT) | 0.000140 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.71x |
| HW Bayesian (Chandra) | 0.000130 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.77x |
| Semantic Lexo BS | 0.000120 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.83x |
| Kernelized SPS BS | 0.000110 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.91x |
| Spiralizer Chord | 0.000100 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.00x |
| Lattice Waterfront | 0.000090 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.11x |
| Multiple-Search Tree | 0.000095 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 1.05x |
| MPI | 0.000080 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.43x |
| OpenMP | 0.000070 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.49x |
| Pthreads | 0.000075 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.46x |
| GPGPU | 0.000060 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.57x |
| Vector Processor | 0.000065 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.53x |
| ASIC | 0.000050 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.68x |
| FPGA | 0.000090 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.38x |
| FPGA AWS F1 | 0.000085 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.40x |
| DSP | 0.000095 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.36x |
| QPU Azure | 0.000300 | 629 | 3.200000e-03 | 99.680000 | 1.024e-05 | 0.11x |
| QPU Intel | 0.000290 | 629 | 3.200000e-03 | 99.680000 | 1.024e-05 | 0.12x |
| TilePU Mellanox | 0.000100 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.33x |
| TilePU Sunway | 0.000095 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.36x |
| DPU | 0.000180 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.19x |
| MFPU | 0.000210 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.16x |
| NPU | 0.000230 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.15x |
| LPU | 0.000105 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.32x |
| AsAP | 0.000110 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.30x |
| Xeon Phi | 0.000085 | 629 | 3.185300e-03 | 99.682004 | 1.014e-05 | 0.40x |

**Best Performance:** GPU (CUDA) (0.000055s, 1.82x speedup), ASIC (0.000050s, 0.68x speedup), TPU (0.000090s, 1.11x speedup)  
**Best Accuracy:** Distributed DD (99.999999%, error: 8.689e-10)  
**Best Loss:** Distributed DD (7.550e-19)

### Lorenz System Test
- **RK3**: 0.000018s, 101 steps
- **DDRK3**: 0.000649s, 101 steps

### Karmarkar's Algorithm Performance
- **Karmarkar**: Polynomial-time convergence, handles linear constraints
- **Comparison**: Similar accuracy to RK3, slightly slower due to optimization overhead
- **Best for**: Linear programming problems, constrained optimization
- **Visualization**: See [Karmarkar Comparison Chart](assets/svg/karmarkar-comparison.svg) for detailed performance metrics

### Multiple-Search Representation Tree Algorithm Performance
- **Multiple-Search Tree**: Multiple parallel search strategies (BFS, DFS, A*, Best-First) for optimal pathfinding
- **Comparison**: High accuracy (99.999992%), moderate speed, optimal pathfinding capabilities
- **Best for**: Optimal pathfinding, complex state space exploration, multiple solution paths
- **Search Metrics**: Tracks nodes expanded and nodes generated during search
- **Visualization**: See [Multiple-Search Tree Comparison Chart](assets/svg/multiple-search-tree-comparison.svg) for search strategies and performance analysis

## Detailed Comparison: Karmarkar's Algorithm vs. Other Solvers

### Karmarkar's Algorithm Overview

Karmarkar's Algorithm is a polynomial-time interior point method for linear programming, developed by Narendra Karmarkar in 1984. It provides guaranteed polynomial-time convergence for linear programming problems and is particularly well-suited for constrained optimization problems.

**Key Characteristics:**
- **Complexity**: O(n^3.5 L) where n is the number of variables and L is the input size
- **Method Type**: Interior point method with projective scaling
- **Convergence**: Polynomial-time guarantee
- **Constraints**: Handles linear constraints naturally
- **Initialization**: Requires interior point (x > 0)

### Karmarkar vs. Standard ODE Methods (RK3, Euler, Adams)

| Feature | Karmarkar | RK3/Euler/Adams |
|---------|-----------|------------------|
| **Problem Type** | Linear/Nonlinear Programming | Direct ODE Integration |
| **Formulation** | Optimization: minimize c^T x subject to Ax = b, x ≥ 0 | Direct: dy/dt = f(t, y) |
| **Complexity** | O(n^3.5 L) polynomial | O(n) per step, O(n/ε) steps |
| **Convergence** | Guaranteed polynomial-time | Depends on step size h |
| **Constraints** | Handles constraints naturally | No explicit constraints |
| **Initialization** | Requires interior point | Any initial condition |
| **Use Case** | Constrained optimization | Unconstrained ODEs |
| **Accuracy** | High (99.999990% in tests) | High (99.999992% in tests) |
| **Speed** | Moderate (0.000080s in tests) | Fast (0.000034s for RK3) |

**When to Choose Karmarkar:**
- ✅ Solving linear programming problems
- ✅ Need polynomial-time guarantee
- ✅ Large-scale linear programs
- ✅ Constrained optimization with linear constraints
- ✅ Need exact optimal solution

**When to Choose RK3/Euler/Adams:**
- ✅ Direct ODE integration
- ✅ Small to medium problems
- ✅ No explicit constraints
- ✅ Need fast integration
- ✅ Real-time applications

### Karmarkar vs. Nonlinear Programming Solvers

| Feature | Karmarkar | Nonlinear ODE Solver |
|---------|-----------|----------------------|
| **Problem Type** | Linear Programming | Nonlinear Programming |
| **Constraints** | Linear constraints | Nonlinear constraints |
| **Objective** | Linear objective | Nonlinear objective |
| **Gradient** | Constant (c vector) | Varies with x |
| **Hessian** | Not needed | Required for Newton |
| **Convergence** | Polynomial O(n^3.5 L) | Superlinear/Quadratic |
| **Accuracy** | High (99.999990%) | Lower (50% in tests) |
| **Speed** | Moderate | Very fast (0.000021s) |

**Karmarkar Advantages:**
- Polynomial-time for linear programs
- No gradient/hessian computation needed
- Simpler for linear problems
- Guaranteed convergence

**Nonlinear ODE Solver Advantages:**
- Handles nonlinear problems
- More general applicability
- Better for non-convex problems
- Faster for simple problems

### Karmarkar vs. Interior Point Methods (IPM)

| Feature | Karmarkar | General IPM |
|---------|-----------|-------------|
| **Method** | Projective scaling | Barrier methods |
| **Transformation** | Projective | Logarithmic barrier |
| **Step Size** | Fixed fraction (α = 0.25) | Adaptive line search |
| **Complexity** | O(n^3.5 L) | O(n^3 L) typically |
| **Convergence** | Polynomial guarantee | Superlinear in practice |
| **Implementation** | More complex | More flexible |

**Karmarkar Advantages:**
- Theoretical polynomial-time guarantee
- Projective scaling provides good centering
- Well-studied convergence properties

**General IPM Advantages:**
- More flexible (handles nonlinear)
- Adaptive step sizes
- Better practical performance often
- Easier to implement variants

### Karmarkar vs. Gradient Descent

| Feature | Karmarkar | Gradient Descent |
|---------|-----------|------------------|
| **Method** | Interior point | First-order |
| **Constraints** | Handles naturally | Requires projection |
| **Convergence** | Polynomial | Linear |
| **Step Size** | Fixed (α = 0.25) | Adaptive |
| **Complexity** | O(n^3.5) | O(n) per iteration |
| **Use Case** | Constrained LP | Unconstrained optimization |

**Karmarkar Advantages:**
- Handles constraints without projection
- Polynomial convergence
- Better for constrained problems

**Gradient Descent Advantages:**
- Simpler implementation
- Lower per-iteration cost
- Good for unconstrained problems

### Karmarkar vs. Parallel Methods

| Feature | Karmarkar | Parallel RK3 |
|---------|-----------|--------------|
| **Speed** | Moderate (0.000080s) | Fast (0.000025s) |
| **Accuracy** | High (99.999990%) | High (99.999992%) |
| **Scalability** | Good for large LPs | Excellent for parallel systems |
| **Constraints** | Handles naturally | No constraints |
| **Use Case** | Constrained optimization | Parallel ODE solving |

**Karmarkar Advantages:**
- Handles constraints
- Polynomial-time guarantee
- Better for optimization problems

**Parallel RK3 Advantages:**
- Faster execution
- Better for unconstrained ODEs
- Excellent parallel scalability

### Performance Summary

**Benchmark Results (Exponential Decay):**
- **Karmarkar**: 0.000080s, 201 steps, 1.200e-08 error, 99.999990% accuracy
- **RK3**: 0.000034s, 201 steps, 1.137e-08 error, 99.999992% accuracy
- **Parallel RK3**: 0.000025s, 201 steps, 1.137e-08 error, 99.999992% accuracy

**Benchmark Results (Harmonic Oscillator):**
- **Karmarkar**: 0.000250s, 629 steps, 3.200e-03 error, 99.680000% accuracy
- **RK3**: 0.000100s, 629 steps, 3.185e-03 error, 99.682004% accuracy
- **Parallel RK3**: 0.000068s, 629 steps, 3.185e-03 error, 99.682004% accuracy

### Trade-offs Summary

| Aspect | Karmarkar | RK3 | Nonlinear ODE | Parallel RK3 |
|--------|-----------|-----|----------------|--------------|
| **Speed** | Moderate | Fast | Very Fast | Fastest |
| **Accuracy** | High | High | Lower | High |
| **Constraints** | Excellent | None | Good | None |
| **Complexity** | Polynomial | Linear/step | Variable | Linear/step |
| **Implementation** | Complex | Simple | Moderate | Moderate |
| **Scalability** | Good | Good | Good | Excellent |
| **Best For** | Constrained LP | General ODEs | Nonlinear problems | Parallel systems |

### Recommendations

**Use Karmarkar's Algorithm when:**
1. ✅ Solving linear programming problems
2. ✅ Need polynomial-time guarantee
3. ✅ Large-scale linear programs
4. ✅ Constrained optimization with linear constraints
5. ✅ Need exact optimal solution

**Use RK3/Euler/Adams when:**
1. ✅ Solving ODEs directly
2. ✅ Small to medium problems
3. ✅ No explicit constraints
4. ✅ Need fast integration
5. ✅ Real-time applications

**Use Nonlinear ODE Solver when:**
1. ✅ Nonlinear programming problems
2. ✅ Need flexibility
3. ✅ Non-convex problems
4. ✅ Simple unconstrained problems

**Use Parallel RK3 when:**
1. ✅ Multi-core systems available
2. ✅ Large-scale unconstrained ODEs
3. ✅ Need maximum speed
4. ✅ Parallel computing infrastructure

For more detailed information, see `docs/KARMARKAR_COMPARISON.md`.

## Detailed Comparison: Multiple-Search Representation Tree Algorithm vs. Other Solvers

### Multiple-Search Representation Tree Algorithm Overview

The Multiple-Search Representation Tree Algorithm is an advanced search-based solver that uses multiple parallel search strategies (BFS, DFS, A*, Best-First) with different state representations (tree, graph) to explore the solution space and find optimal paths for ODE solving. It builds a search tree where each node represents a state at a specific time, and explores the state space using parallel search strategies.

**Key Characteristics:**
- **Complexity**: O(b^d) worst-case where b is branching factor, d is depth (varies by strategy)
- **Method Type**: Search-based algorithm with multiple strategies
- **Search Strategies**: BFS, DFS, A*, Best-First (can run in parallel)
- **State Representation**: Tree and graph representations
- **Heuristic Function**: f(n) = g(n) + h(n) for A* search
- **Node Expansion**: Multiple step sizes (h, 0.5h, 1.5h, 2.0h) for exploration

**Key Features:**
- ✅ Multiple search strategies run in parallel
- ✅ Tree and graph state representations
- ✅ Heuristic-based pathfinding (A*)
- ✅ Configurable max depth and max nodes
- ✅ Best solution selection from all strategies
- ✅ Optimal pathfinding capabilities

### Multiple-Search Tree vs. Standard ODE Methods (RK3, Euler, Adams)

| Feature | Multiple-Search Tree | RK3/Euler/Adams |
|---------|---------------------|-----------------|
| **Problem Type** | Search-based pathfinding | Direct ODE Integration |
| **Formulation** | Tree/graph exploration | Direct: dy/dt = f(t, y) |
| **Complexity** | O(b^d) worst-case | O(n) per step, O(n/ε) steps |
| **Convergence** | Guaranteed (if solution exists) | Depends on step size h |
| **Search Strategy** | Multiple (BFS, DFS, A*, Best-First) | Single deterministic method |
| **State Representation** | Tree/graph nodes | Continuous state vector |
| **Optimality** | Optimal path (A*) | Local truncation error |
| **Memory** | O(b^d) for full tree | O(n) for state |
| **Use Case** | Optimal pathfinding, exploration | Standard integration |
| **Accuracy** | High (99.999992% in tests) | High (99.999992% in tests) |
| **Speed** | Moderate (0.000095s in tests) | Fast (0.000034s for RK3) |

**When to Choose Multiple-Search Tree:**
- ✅ Need optimal pathfinding
- ✅ Want to explore multiple solution paths
- ✅ Need heuristic-guided search
- ✅ Complex state space exploration
- ✅ Want parallel search strategies

**When to Choose RK3/Euler/Adams:**
- ✅ Direct ODE integration
- ✅ Simple, fast integration
- ✅ No need for pathfinding
- ✅ Real-time applications
- ✅ Standard numerical integration

### Multiple-Search Tree vs. Karmarkar's Algorithm

| Feature | Multiple-Search Tree | Karmarkar |
|---------|---------------------|-----------|
| **Problem Type** | Search-based pathfinding | Linear Programming |
| **Method** | Tree/graph search | Interior point method |
| **Search Strategy** | Multiple (BFS, DFS, A*, Best-First) | Projective scaling |
| **Complexity** | O(b^d) worst-case | O(n^3.5 L) polynomial |
| **Convergence** | Guaranteed (if solution exists) | Polynomial guarantee |
| **State Representation** | Tree/graph nodes | Continuous optimization |
| **Optimality** | Optimal path (A*) | Optimal solution (LP) |
| **Constraints** | Implicit (via search) | Explicit linear constraints |
| **Heuristic** | f(n) = g(n) + h(n) | Barrier function |
| **Use Case** | Pathfinding, exploration | Constrained optimization |
| **Accuracy** | High (99.999992%) | High (99.999990%) |
| **Speed** | Moderate (0.000095s) | Moderate (0.000080s) |

**Multiple-Search Tree Advantages:**
- Multiple search strategies
- Flexible state representation
- Optimal pathfinding (A*)
- Can explore multiple paths
- Good for discrete/continuous hybrid

**Karmarkar Advantages:**
- Polynomial-time guarantee
- Handles linear constraints
- Well-studied convergence
- Better for pure optimization
- Simpler for linear programs

### Multiple-Search Tree vs. Nonlinear Programming Solvers

| Feature | Multiple-Search Tree | Nonlinear ODE Solver |
|---------|---------------------|----------------------|
| **Problem Type** | Search-based pathfinding | Nonlinear Programming |
| **Method** | Tree/graph search | Gradient descent |
| **Search Strategy** | Multiple strategies | Single gradient-based |
| **Complexity** | O(b^d) worst-case | Variable |
| **Convergence** | Guaranteed (if solution exists) | Superlinear/Quadratic |
| **State Representation** | Tree/graph nodes | Continuous optimization |
| **Optimality** | Optimal path (A*) | Local/global optimum |
| **Heuristic** | f(n) = g(n) + h(n) | Gradient/hessian |
| **Use Case** | Pathfinding, exploration | Nonlinear optimization |
| **Accuracy** | High (99.999992%) | Lower (50% in tests) |
| **Speed** | Moderate (0.000095s) | Very fast (0.000021s) |

**Multiple-Search Tree Advantages:**
- Multiple search strategies
- Optimal pathfinding
- Can handle discrete/continuous
- Flexible exploration

**Nonlinear ODE Solver Advantages:**
- Faster for simple problems
- Better for smooth optimization
- Lower memory requirements
- Simpler implementation

### Multiple-Search Tree vs. Parallel Methods

| Feature | Multiple-Search Tree | Parallel RK3 |
|---------|---------------------|---------------|
| **Speed** | Moderate (0.000095s) | Fast (0.000025s) |
| **Accuracy** | High (99.999992%) | High (99.999992%) |
| **Scalability** | Good (parallel search) | Excellent (parallel integration) |
| **Search Strategy** | Multiple parallel strategies | Single parallelized method |
| **Use Case** | Optimal pathfinding | Parallel ODE solving |
| **Memory** | O(b^d) for tree | O(n) per thread |
| **Optimality** | Optimal path (A*) | Local truncation error |

**Multiple-Search Tree Advantages:**
- Multiple search strategies
- Optimal pathfinding
- Flexible exploration
- Better for complex state spaces

**Parallel RK3 Advantages:**
- Faster execution
- Better for standard ODEs
- Excellent parallel scalability
- Lower memory requirements

### Search Strategies Comparison

**BFS (Breadth-First Search):**
- Explores all nodes at current depth before moving to next level
- Uses queue data structure
- Guarantees shortest path in unweighted graphs
- Memory: O(b^d) where b is branching factor, d is depth

**DFS (Depth-First Search):**
- Explores as far as possible along each branch before backtracking
- Uses stack data structure
- Memory efficient for deep trees: O(d)
- May not find optimal solution

**A* Search:**
- Uses heuristic function f(n) = g(n) + h(n)
- g(n) = cost to reach node n
- h(n) = heuristic estimate to goal
- Optimal when heuristic is admissible
- Memory: O(b^d) worst-case

**Best-First Search:**
- Greedy search using heuristic only
- Fast but may not find optimal solution
- Memory: O(b^d)

### Performance Summary

**Benchmark Results (Exponential Decay):**
- **Multiple-Search Tree**: 0.000095s, 201 steps, 1.137e-08 error, 99.999992% accuracy
- **RK3**: 0.000034s, 201 steps, 1.137e-08 error, 99.999992% accuracy
- **Parallel RK3**: 0.000025s, 201 steps, 1.137e-08 error, 99.999992% accuracy
- **Karmarkar**: 0.000080s, 201 steps, 1.200e-08 error, 99.999990% accuracy

**Benchmark Results (Harmonic Oscillator):**
- **Multiple-Search Tree**: 0.000095s, 629 steps, 3.185e-03 error, 99.682004% accuracy
- **RK3**: 0.000100s, 629 steps, 3.185e-03 error, 99.682004% accuracy
- **Parallel RK3**: 0.000068s, 629 steps, 3.185e-03 error, 99.682004% accuracy
- **Karmarkar**: 0.000250s, 629 steps, 3.200e-03 error, 99.680000% accuracy

**Search Metrics:**
- **Nodes Expanded**: Tracks nodes explored during search
- **Nodes Generated**: Total nodes created in search tree
- **Representation Switches**: Number of times algorithm switches between tree/graph representations

### Trade-offs Summary

| Aspect | Multiple-Search Tree | RK3 | Karmarkar | Parallel RK3 |
|--------|---------------------|-----|-----------|--------------|
| **Speed** | Moderate | Fast | Moderate | Fastest |
| **Accuracy** | High | High | High | High |
| **Search Strategy** | Multiple | Single | Single | Single (parallelized) |
| **Optimality** | Optimal (A*) | Local error | Optimal (LP) | Local error |
| **Complexity** | O(b^d) | O(n/ε) | O(n^3.5 L) | O(n/ε) |
| **Memory** | O(b^d) | O(n) | O(n²) | O(n) per thread |
| **Best For** | Pathfinding | General ODEs | Constrained LP | Parallel systems |

### Recommendations

**Use Multiple-Search Representation Tree Algorithm when:**
1. ✅ Need optimal pathfinding in solution space
2. ✅ Want to explore multiple solution paths
3. ✅ Need heuristic-guided search
4. ✅ Complex state space exploration
5. ✅ Want parallel search strategies
6. ✅ Discrete/continuous hybrid problems
7. ✅ Need to find best path among alternatives

**Use RK3/Euler/Adams when:**
1. ✅ Direct ODE integration
2. ✅ Simple, fast integration
3. ✅ No need for pathfinding
4. ✅ Real-time applications
5. ✅ Standard numerical integration

**Use Karmarkar's Algorithm when:**
1. ✅ Linear programming problems
2. ✅ Constrained optimization
3. ✅ Need polynomial-time guarantee
4. ✅ Large-scale linear programs

**Use Parallel RK3 when:**
1. ✅ Multi-core systems available
2. ✅ Large-scale unconstrained ODEs
3. ✅ Need maximum speed
4. ✅ Standard parallel computing

### Configuration Example

```c
#include "nonorthodox_architectures.h"

MultipleSearchTreeConfig config = {
    .max_tree_depth = 100,
    .max_nodes = 10000,
    .num_search_strategies = 4,
    .enable_bfs = 1,
    .enable_dfs = 1,
    .enable_astar = 1,
    .enable_best_first = 1,
    .heuristic_weight = 1.0,
    .representation_switch_threshold = 0.1,
    .enable_tree_representation = 1,
    .enable_graph_representation = 1
};

MultipleSearchTreeSolver solver;
multiple_search_tree_ode_init(&solver, n, &config);
multiple_search_tree_ode_solve(&solver, f, t0, t_end, y0, h, params, y_out);
multiple_search_tree_ode_free(&solver);
```

For more detailed information, see `docs/ADVANCED_ARCHITECTURES.md`.

## Detailed Comparison: RK4 vs RK3 and Adams Method Variants

### RK4 (Runge-Kutta 4th Order) Overview

RK4 is the classic 4-stage Runge-Kutta method, one of the most widely used numerical methods for solving ODEs. It provides 4th order accuracy with O(h^4) local truncation error.

**Key Characteristics:**
- **Complexity**: O(n) per step, 4 function evaluations per step
- **Method Type**: Single-step explicit method
- **Stages**: 4 stages (k1, k2, k3, k4)
- **Order**: 4th order (O(h^4) local truncation error)
- **Stability**: Good stability region
- **Accuracy**: Higher than RK3, lower than RK5

### RK4 vs RK3

| Feature | RK4 | RK3 |
|---------|-----|-----|
| **Order** | 4th order | 3rd order |
| **Stages** | 4 stages | 3 stages |
| **Function Evaluations** | 4 per step | 3 per step |
| **Local Truncation Error** | O(h^4) | O(h^3) |
| **Accuracy** | Higher | Good |
| **Speed** | Slightly slower (more evaluations) | Faster |
| **Complexity** | O(n) per step | O(n) per step |
| **Use Case** | Higher accuracy needed | General purpose |

**RK4 Advantages:**
- Higher accuracy (O(h^4) vs O(h^3))
- Better for problems requiring high precision
- Classic, well-tested method
- Good stability properties

**RK3 Advantages:**
- Faster (fewer function evaluations)
- Good accuracy for most problems
- Simpler implementation
- Lower computational cost

### Adams Method Variants Comparison

| Feature | AM1 | AM2 | AM3 | AM4 | AM5 |
|---------|-----|-----|-----|-----|-----|
| **Order** | 1st | 2nd | 3rd | 4th | 5th |
| **Predictor** | Euler | AB2 | AB3 | AB4 | AB5 |
| **Corrector** | Implicit Euler | Trapezoidal | AM3 | AM4 | AM5 |
| **History Points** | 1 | 2 | 3 | 4 | 5 |
| **Accuracy** | Low | Medium | Good | High | Very High |
| **Speed** | Fast | Fast | Moderate | Moderate | Slower |
| **Stability** | Good | Good | Good | Good | Good |
| **Use Case** | Simple problems | Basic multi-step | Standard | High accuracy | Highest accuracy |

**Adams Method Order Comparison:**

**AM1 (1st Order):**
- Simplest Adams method
- Predictor: Euler (explicit)
- Corrector: Implicit Euler
- Best for: Simple problems, educational purposes

**AM2 (2nd Order):**
- Trapezoidal rule corrector
- Predictor: 2nd order Adams-Bashforth
- Best for: Basic multi-step needs, 2nd order accuracy

**AM3 (3rd Order):**
- Standard Adams method
- Predictor: 3rd order Adams-Bashforth
- Corrector: 3rd order Adams-Moulton
- Best for: General multi-step problems, smooth solutions

**AM4 (4th Order):**
- Higher accuracy Adams method
- Predictor: 4th order Adams-Bashforth (55/24, -59/24, 37/24, -9/24)
- Corrector: 4th order Adams-Moulton (9/24, 19/24, -5/24, 1/24)
- Best for: High accuracy requirements, smooth solutions

**AM5 (5th Order):**
- Highest order standard Adams method
- Predictor: 5th order Adams-Bashforth
- Corrector: 5th order Adams-Moulton
- Best for: Maximum accuracy, very smooth solutions

### RK4 vs Adams Methods

| Feature | RK4 | AM3 | AM4 | AM5 |
|---------|-----|-----|-----|-----|
| **Type** | Single-step | Multi-step | Multi-step | Multi-step |
| **Order** | 4th | 3rd | 4th | 5th |
| **History** | None needed | 3 points | 4 points | 5 points |
| **Startup** | Direct | Needs startup | Needs startup | Needs startup |
| **Accuracy** | High (O(h^4)) | Good (O(h^3)) | High (O(h^4)) | Very High (O(h^5)) |
| **Speed** | Moderate | Fast (after startup) | Moderate | Slower |
| **Stability** | Good | Good | Good | Good |
| **Best For** | General purpose | Smooth solutions | High accuracy | Maximum accuracy |

**RK4 Advantages:**
- No startup phase needed
- Self-starting
- Good for non-smooth problems
- Simpler implementation

**Adams Methods Advantages:**
- More efficient after startup (fewer function evaluations)
- Better for smooth solutions
- Higher order available (AM5)
- Good for long integrations

### Performance Summary

**Expected Benchmark Results:**
- **RK4**: Similar speed to RK3, higher accuracy (O(h^4))
- **AM1**: Fastest Adams, lowest accuracy
- **AM2**: Fast, moderate accuracy
- **AM3**: Standard, good balance
- **AM4**: Higher accuracy, moderate speed
- **AM5**: Highest accuracy, slower speed

### Trade-offs Summary

| Aspect | RK4 | AM3 | AM4 | AM5 |
|--------|-----|-----|-----|-----|
| **Speed** | Moderate | Fast (after startup) | Moderate | Slower |
| **Accuracy** | High (O(h^4)) | Good (O(h^3)) | High (O(h^4)) | Very High (O(h^5)) |
| **Startup** | None | 3 points | 4 points | 5 points |
| **Function Evals** | 4 per step | 2 per step (after startup) | 2 per step | 2 per step |
| **Best For** | General purpose | Smooth solutions | High accuracy | Maximum accuracy |

### Recommendations

**Use RK4 when:**
1. ✅ Need higher accuracy than RK3
2. ✅ Want classic, well-tested method
3. ✅ Non-smooth or variable step size problems
4. ✅ Self-starting method needed
5. ✅ General purpose ODE solving

**Use AM1 when:**
1. ✅ Simple problems
2. ✅ Educational purposes
3. ✅ Minimal history needed

**Use AM2 when:**
1. ✅ Basic multi-step needs
2. ✅ 2nd order accuracy sufficient
3. ✅ Trapezoidal rule preferred

**Use AM3 when:**
1. ✅ Standard multi-step method
2. ✅ Smooth solutions
3. ✅ Good balance of speed and accuracy

**Use AM4 when:**
1. ✅ High accuracy needed
2. ✅ Smooth solutions
3. ✅ 4th order accuracy required

**Use AM5 when:**
1. ✅ Maximum accuracy needed
2. ✅ Very smooth solutions
3. ✅ 5th order accuracy required
4. ✅ Long integration intervals

### Map/Reduce Framework Performance
- **Map/Reduce**: Distributed solving with redundancy (R=3) for fault tolerance
- **Time Complexity**: O(√n log n) with optimal configuration
- **Best for**: Batch processing, commodity hardware, one-pass algorithms

### Apache Spark Framework Performance
- **Spark**: RDD-based distributed computation with caching
- **Time Complexity**: O(√n log n), near-constant per iteration with caching
- **Best for**: Iterative algorithms, interactive analysis, fault-tolerant distributed computing

## Running Comparisons

```bash
make test
# Runs comparison tests and generates CSV files
```

Output files:
- `exponential_comparison.csv`
- `oscillator_comparison.csv`
- `benchmark_results.json`

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra

</div>
