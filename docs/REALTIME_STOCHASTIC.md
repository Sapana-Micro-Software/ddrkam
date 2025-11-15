# Real-Time and Stochastic Solvers Guide

## Overview

DDRKAM now includes **real-time** and **stochastic** variants of RK3 and Adams methods, along with **data-driven** adaptive control for solving differential equations in streaming and uncertain environments.

## Features

### 1. Real-Time Solvers
- **Streaming data processing**: Process ODE solutions as data arrives
- **Buffer management**: Optional buffering for historical data
- **Callback support**: Real-time notifications for each step
- **Low latency**: Optimized for real-time applications

### 2. Stochastic Solvers
- **Noise injection**: Add stochastic noise to ODE solutions
- **Brownian motion**: Support for Brownian motion processes
- **White noise**: Gaussian white noise option
- **Configurable parameters**: Adjustable noise amplitude and correlation

### 3. Data-Driven Methods
- **Adaptive step size**: Automatically adjust step size based on error history
- **Method selection**: Choose optimal method (RK3 vs Adams) based on system characteristics
- **Performance optimization**: Balance accuracy and speed requirements

## C/C++ API Usage

### Real-Time RK3 Solver

```c
#include "realtime_stochastic.h"

// Initialize real-time solver
RealtimeSolverState state;
realtime_rk3_init(&state, 1, 0.01, 10); // dimension=1, h=0.01, buffer=10

// Streaming callback
void callback(double t, const double* y, size_t n, void* user_data) {
    printf("t=%.3f, y=%.6f\n", t, y[0]);
}

// Process streaming data
double y_new[1] = {1.0};
for (int i = 0; i < 100; i++) {
    y_new[0] = get_streaming_data(); // Get new data point
    realtime_rk3_step(&state, my_ode, y_new, NULL, callback, NULL);
}

realtime_solver_free(&state);
```

### Stochastic RK3 Solver

```c
// Configure stochastic parameters
StochasticParams params = {
    .noise_amplitude = 0.01,
    .noise_correlation = 0.1,
    .use_brownian = 0,  // 0 = white noise, 1 = Brownian motion
    .seed = 42
};

// Initialize stochastic solver
void* solver = stochastic_rk3_init(1, 0.01, &params);

// Solve with noise
double y[1] = {1.0};
double t = 0.0;
for (int i = 0; i < 100; i++) {
    t = stochastic_rk3_step(solver, my_ode, t, y, NULL);
    printf("t=%.3f, y=%.6f\n", t, y[0]);
}

stochastic_solver_free(solver);
```

### Data-Driven Adaptive Control

```c
// Adaptive step size based on error history
double errors[] = {1e-3, 5e-4, 2e-4, 1e-4, 5e-5};
double current_h = 0.01;
double target_error = 1e-4;

double new_h = data_driven_adaptive_step(errors, 5, current_h, target_error);
printf("Recommended step size: %.6f\n", new_h);

// Method selection
double stiffness = 200.0;
double tolerance = 1e-4;
double speed_req = 500000.0;

int method = data_driven_method_select(stiffness, tolerance, speed_req);
// Returns: 0 = RK3, 1 = Adams, -1 = use both
```

## Objective-C API Usage

```objc
#import <DDRKAM/DDRKAMRealtimeStochastic.h>

// Real-Time Solver
DDRKAMRealtimeSolver* realtimeSolver = 
    [[DDRKAMRealtimeSolver alloc] initWithDimension:2 
                                            stepSize:0.01 
                                          bufferSize:10];

[realtimeSolver stepRK3WithFunction:^(double t, const double* y, double* dydt, void* params) {
    dydt[0] = -y[0];
    dydt[1] = -2.0 * y[1];
} newState:@[@1.0, @1.0] params:NULL callback:^(double t, NSArray<NSNumber*>* y) {
    NSLog(@"t=%.3f, y=[%.6f, %.6f]", t, [y[0] doubleValue], [y[1] doubleValue]);
}];

// Stochastic Solver
DDRKAMStochasticParams* params = 
    [DDRKAMStochasticParams paramsWithAmplitude:0.01 useBrownian:NO];

DDRKAMStochasticSolver* stochasticSolver = 
    [[DDRKAMStochasticSolver alloc] initRK3WithDimension:1 
                                                stepSize:0.01 
                                                  params:params];

NSMutableArray<NSNumber*>* y = [NSMutableArray arrayWithObject:@1.0];
double t = 0.0;
for (int i = 0; i < 100; i++) {
    t = [stochasticSolver stepRK3WithFunction:^(double t, const double* y, double* dydt, void* p) {
        dydt[0] = -y[0];
    } time:t state:y params:NULL];
}

// Data-Driven Control
NSArray<NSNumber*>* errors = @[@1e-3, @5e-4, @2e-4, @1e-4, @5e-5];
double newH = [DDRKAMDataDrivenControl adaptiveStepSizeWithErrorHistory:errors 
                                                               currentH:0.01 
                                                            targetError:1e-4];

NSInteger method = [DDRKAMDataDrivenControl selectMethodWithStiffness:200.0 
                                                         errorTolerance:1e-4 
                                                        speedRequirement:500000.0];
```

## Applications

### Real-Time Solvers
- **Sensor data processing**: Real-time analysis of streaming sensor data
- **Control systems**: Online control and monitoring
- **Live simulations**: Interactive simulations with user input
- **Data acquisition**: Processing data as it arrives

### Stochastic Solvers
- **Uncertainty quantification**: Modeling systems with uncertain parameters
- **Noise analysis**: Studying system behavior under noise
- **Monte Carlo methods**: Stochastic simulations
- **Robustness testing**: Testing system robustness to perturbations

### Data-Driven Methods
- **Adaptive control**: Automatic parameter adjustment
- **Performance optimization**: Balancing accuracy and speed
- **System identification**: Learning optimal solver parameters
- **Resource management**: Efficient computational resource usage

## Copyright

Copyright (C) 2025, Shyamal Suhana Chandra
