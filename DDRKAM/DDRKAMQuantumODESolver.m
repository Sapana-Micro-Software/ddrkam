/*
 * DDRKAM Quantum ODE Solver - Objective-C Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMQuantumODESolver.h"
#import "quantum_ode_solver.h"
#import <Foundation/Foundation.h>

// ============================================================================
// Quantum Parameters
// ============================================================================

@implementation DDRKAMQuantumParams

- (instancetype)initWithTemperature:(double)temperature
                   tunnelingStrength:(double)tunnelingStrength
                      coherenceTime:(double)coherenceTime
                       numIterations:(NSUInteger)numIterations
              convergenceThreshold:(double)convergenceThreshold {
    self = [super init];
    if (self) {
        _temperature = temperature;
        _tunnelingStrength = tunnelingStrength;
        _coherenceTime = coherenceTime;
        _numIterations = numIterations;
        _convergenceThreshold = convergenceThreshold;
    }
    return self;
}

- (instancetype)init {
    return [self initWithTemperature:1.0
                    tunnelingStrength:0.1
                       coherenceTime:1.0
                        numIterations:100
               convergenceThreshold:1e-6];
}

@end

// ============================================================================
// Quantum ODE Solver
// ============================================================================

@implementation DDRKAMQuantumODESolver {
    QuantumODESolver _solver;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                      numQuantumStates:(NSUInteger)numQuantumStates
                          historySize:(NSUInteger)historySize
                     predictionHorizon:(NSUInteger)predictionHorizon
                    nonlinearODEFunction:(void(*)(double t, const double* y, double* dydt,
                                                  const double* y_history, size_t history_size,
                                                  void* params))nonlinearODEFunc
                              odeParams:(void* _Nullable)odeParams
                         quantumParams:(DDRKAMQuantumParams* _Nullable)quantumParams
                                 error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    QuantumParams c_quantum_params;
    if (quantumParams) {
        c_quantum_params.temperature = quantumParams.temperature;
        c_quantum_params.tunneling_strength = quantumParams.tunnelingStrength;
        c_quantum_params.coherence_time = quantumParams.coherenceTime;
        c_quantum_params.num_iterations = (size_t)quantumParams.numIterations;
        c_quantum_params.convergence_threshold = quantumParams.convergenceThreshold;
    } else {
        c_quantum_params.temperature = 1.0;
        c_quantum_params.tunneling_strength = 0.1;
        c_quantum_params.coherence_time = 1.0;
        c_quantum_params.num_iterations = 100;
        c_quantum_params.convergence_threshold = 1e-6;
    }
    
    int result = quantum_ode_init(&_solver, stateDim, stepSize, numQuantumStates,
                                  historySize, predictionHorizon,
                                  nonlinearODEFunc, odeParams,
                                  quantumParams ? &c_quantum_params : NULL);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Quantum ODE initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    quantum_ode_free(&_solver);
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (double)stepSize {
    return _solver.step_size;
}

- (NSUInteger)numQuantumStates {
    return _solver.num_quantum_states;
}

- (NSUInteger)predictionHorizon {
    return _solver.prediction_horizon;
}

- (double)currentTime {
    return _solver.current_time;
}

- (NSUInteger)totalSteps {
    return _solver.total_steps;
}

- (uint64_t)quantumOperations {
    return _solver.quantum_operations;
}

- (double)predictionConfidence {
    return _solver.prediction_confidence;
}

- (BOOL)usePostRealtime {
    return _solver.use_post_realtime != 0;
}

- (void)setUsePostRealtime:(BOOL)usePostRealtime {
    _solver.use_post_realtime = usePostRealtime ? 1 : 0;
}

- (BOOL)stepAtTime:(double)t
              state:(NSArray<NSNumber*>*)y
              error:(NSError**)error {
    if (!y || y.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid state dimension"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y[i] = [y[i] doubleValue];
    }
    
    int result = quantum_ode_step(&_solver, t, c_y);
    
    if (result == 0) {
        // Update y array (in-place modification)
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            // Note: This modifies the input array
        }
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Quantum ODE step failed"}];
    }
    
    free(c_y);
    return result == 0;
}

- (BOOL)predictFutureAtTime:(double)t
                 currentState:(NSArray<NSNumber*>*)yCurrent
                futureStates:(NSArray<NSArray<NSNumber*>*>**)futureStates
                  confidence:(double*)confidence
                       error:(NSError**)error {
    if (!yCurrent || !futureStates || !confidence || yCurrent.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y_current = (double*)malloc(_solver.state_dim * sizeof(double));
    double** c_future = (double**)malloc(_solver.prediction_horizon * sizeof(double*));
    
    if (!c_y_current || !c_future) {
        free(c_y_current);
        if (c_future) {
            for (NSUInteger i = 0; i < _solver.prediction_horizon; i++) {
                if (c_future[i]) free(c_future[i]);
            }
            free(c_future);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y_current[i] = [yCurrent[i] doubleValue];
    }
    
    for (NSUInteger i = 0; i < _solver.prediction_horizon; i++) {
        c_future[i] = (double*)malloc(_solver.state_dim * sizeof(double));
        if (!c_future[i]) {
            for (NSUInteger j = 0; j < i; j++) {
                free(c_future[j]);
            }
            free(c_future);
            free(c_y_current);
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    int result = quantum_ode_predict_future(&_solver, t, c_y_current, c_future, confidence);
    
    if (result == 0) {
        NSMutableArray<NSArray<NSNumber*>*>* future = [NSMutableArray arrayWithCapacity:_solver.prediction_horizon];
        for (NSUInteger i = 0; i < _solver.prediction_horizon; i++) {
            NSMutableArray<NSNumber*>* step = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                [step addObject:@(c_future[i][j])];
            }
            [future addObject:[step copy]];
        }
        *futureStates = [future copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Future prediction failed"}];
    }
    
    for (NSUInteger i = 0; i < _solver.prediction_horizon; i++) {
        free(c_future[i]);
    }
    free(c_future);
    free(c_y_current);
    
    return result == 0;
}

- (BOOL)refineAtTime:(double)t
         initialSolution:(NSArray<NSNumber*>*)yInitial
          refinedSolution:(NSArray<NSNumber*>**)yRefined
         maxIterations:(NSUInteger)maxIterations
                  error:(NSError**)error {
    if (!yInitial || !yRefined || yInitial.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y_initial = (double*)malloc(_solver.state_dim * sizeof(double));
    double* c_y_refined = (double*)malloc(_solver.state_dim * sizeof(double));
    
    if (!c_y_initial || !c_y_refined) {
        free(c_y_initial);
        free(c_y_refined);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y_initial[i] = [yInitial[i] doubleValue];
    }
    
    int result = quantum_ode_refine(&_solver, t, c_y_initial, c_y_refined, maxIterations);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* refined = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            [refined addObject:@(c_y_refined[i])];
        }
        *yRefined = [refined copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Refinement failed"}];
    }
    
    free(c_y_initial);
    free(c_y_refined);
    
    return result == 0;
}

- (BOOL)solveFromTime:(double)t0
               toTime:(double)tEnd
        initialCondition:(NSArray<NSNumber*>*)y0
          numSteps:(NSUInteger)numSteps
      solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
             error:(NSError**)error {
    if (!y0 || !solutionPath || y0.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double** c_solution = (double**)malloc(numSteps * sizeof(double*));
    double* c_y0 = (double*)malloc(_solver.state_dim * sizeof(double));
    
    if (!c_solution || !c_y0) {
        if (c_solution) {
            for (NSUInteger i = 0; i < numSteps; i++) {
                if (c_solution[i]) free(c_solution[i]);
            }
            free(c_solution);
        }
        free(c_y0);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < numSteps; i++) {
        c_solution[i] = (double*)malloc(_solver.state_dim * sizeof(double));
        if (!c_solution[i]) {
            for (NSUInteger j = 0; j < i; j++) {
                free(c_solution[j]);
            }
            free(c_solution);
            free(c_y0);
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    int result = quantum_ode_solve(&_solver, t0, tEnd, c_y0, c_solution, numSteps);
    
    if (result == 0) {
        NSMutableArray<NSArray<NSNumber*>*>* path = [NSMutableArray arrayWithCapacity:numSteps];
        for (NSUInteger i = 0; i < numSteps; i++) {
            NSMutableArray<NSNumber*>* step = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                [step addObject:@(c_solution[i][j])];
            }
            [path addObject:[step copy]];
        }
        *solutionPath = [path copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMQuantumODESolver"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Quantum ODE solve failed"}];
    }
    
    for (NSUInteger i = 0; i < numSteps; i++) {
        free(c_solution[i]);
    }
    free(c_solution);
    free(c_y0);
    
    return result == 0;
}

@end
