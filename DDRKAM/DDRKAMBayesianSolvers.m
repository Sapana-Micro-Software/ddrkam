/*
 * DDRKAM Bayesian ODE Solvers - Objective-C Implementation
 * Real-Time Bayesian and Randomized DP Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMBayesianSolvers.h"
#import "bayesian_ode_solvers.h"
#import <Foundation/Foundation.h>

// ============================================================================
// Forward-Backward Solver
// ============================================================================

@implementation DDRKAMForwardBackwardSolver {
    ForwardBackwardSolver _solver;
    NSArray<NSNumber*>* _stateValues;
}

- (instancetype)initWithStateSpaceSize:(NSUInteger)stateSpaceSize
                           stateValues:(NSArray<NSNumber*>*)stateValues
                      transitionMatrix:(NSArray<NSArray<NSNumber*>*>*)transitionMatrix
                                 prior:(NSArray<NSNumber*>*)prior
            observationNoiseVariance:(double)noiseVariance {
    self = [super init];
    if (!self) return nil;
    
    if (stateValues.count != stateSpaceSize ||
        transitionMatrix.count != stateSpaceSize ||
        prior.count != stateSpaceSize) {
        return nil;
    }
    
    _stateValues = [stateValues copy];
    
    // Convert NSArray to C arrays
    double* c_state_values = (double*)malloc(stateSpaceSize * sizeof(double));
    double** c_transition = (double**)malloc(stateSpaceSize * sizeof(double*));
    double* c_prior = (double*)malloc(stateSpaceSize * sizeof(double));
    
    if (!c_state_values || !c_transition || !c_prior) {
        free(c_state_values);
        free(c_transition);
        free(c_prior);
        return nil;
    }
    
    for (NSUInteger i = 0; i < stateSpaceSize; i++) {
        c_state_values[i] = [stateValues[i] doubleValue];
        c_prior[i] = [prior[i] doubleValue];
        
        c_transition[i] = (double*)malloc(stateSpaceSize * sizeof(double));
        if (!c_transition[i]) {
            for (NSUInteger j = 0; j < i; j++) {
                free(c_transition[j]);
            }
            free(c_state_values);
            free(c_transition);
            free(c_prior);
            return nil;
        }
        
        NSArray<NSNumber*>* row = transitionMatrix[i];
        for (NSUInteger j = 0; j < stateSpaceSize; j++) {
            c_transition[i][j] = [row[j] doubleValue];
        }
    }
    
    int result = forward_backward_init(&_solver, stateSpaceSize, c_state_values,
                                      c_transition, c_prior, noiseVariance);
    
    // Clean up temporary arrays (solver has its own copies)
    for (NSUInteger i = 0; i < stateSpaceSize; i++) {
        free(c_transition[i]);
    }
    free(c_transition);
    free(c_state_values);
    free(c_prior);
    
    if (result != 0) {
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    forward_backward_free(&_solver);
}

- (NSUInteger)stateSpaceSize {
    return _solver.state_space_size;
}

- (double)observationNoiseVariance {
    return _solver.observation_noise_variance;
}

- (double)currentTime {
    return _solver.current_time;
}

- (NSUInteger)currentStep {
    return _solver.current_step;
}

- (BOOL)stepWithObservation:(double)observation error:(NSError**)error {
    int result = forward_backward_step(&_solver, observation);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Forward-Backward step failed"}];
    }
    return result == 0;
}

- (BOOL)computePosteriorWithError:(NSError**)error {
    int result = forward_backward_compute_posterior(&_solver);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Posterior computation failed"}];
    }
    return result == 0;
}

- (BOOL)getStatisticsMean:(double*)mean
                  variance:(double*)variance
            fullPosterior:(NSArray<NSNumber*>**)fullPosterior
                    error:(NSError**)error {
    if (!mean || !variance) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* posterior_array = NULL;
    if (fullPosterior) {
        posterior_array = (double*)malloc(_solver.state_space_size * sizeof(double));
        if (!posterior_array) {
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    int result = forward_backward_get_statistics(&_solver, mean, variance, posterior_array);
    
    if (result == 0) {
        if (fullPosterior && posterior_array) {
            NSMutableArray<NSNumber*>* posterior = [NSMutableArray arrayWithCapacity:_solver.state_space_size];
            for (NSUInteger i = 0; i < _solver.state_space_size; i++) {
                [posterior addObject:@(posterior_array[i])];
            }
            *fullPosterior = [posterior copy];
        }
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Statistics computation failed"}];
    }
    
    if (posterior_array) {
        free(posterior_array);
    }
    
    return result == 0;
}

@end

// ============================================================================
// Viterbi Solver
// ============================================================================

@implementation DDRKAMViterbiSolver {
    ViterbiSolver _solver;
}

- (instancetype)initWithStateSpaceSize:(NSUInteger)stateSpaceSize
                          stateValues:(NSArray<NSNumber*>*)stateValues
                     transitionMatrix:(NSArray<NSArray<NSNumber*>*>*)transitionMatrix
                                prior:(NSArray<NSNumber*>*)prior
             observationNoiseVariance:(double)noiseVariance {
    self = [super init];
    if (!self) return nil;
    
    // Convert to C arrays (similar to Forward-Backward)
    double* c_state_values = (double*)malloc(stateSpaceSize * sizeof(double));
    double** c_transition = (double**)malloc(stateSpaceSize * sizeof(double*));
    double* c_prior = (double*)malloc(stateSpaceSize * sizeof(double));
    
    if (!c_state_values || !c_transition || !c_prior) {
        free(c_state_values);
        free(c_transition);
        free(c_prior);
        return nil;
    }
    
    for (NSUInteger i = 0; i < stateSpaceSize; i++) {
        c_state_values[i] = [stateValues[i] doubleValue];
        c_prior[i] = [prior[i] doubleValue];
        
        c_transition[i] = (double*)malloc(stateSpaceSize * sizeof(double));
        NSArray<NSNumber*>* row = transitionMatrix[i];
        for (NSUInteger j = 0; j < stateSpaceSize; j++) {
            c_transition[i][j] = [row[j] doubleValue];
        }
    }
    
    int result = viterbi_init(&_solver, stateSpaceSize, c_state_values,
                             c_transition, c_prior, noiseVariance);
    
    for (NSUInteger i = 0; i < stateSpaceSize; i++) {
        free(c_transition[i]);
    }
    free(c_transition);
    free(c_state_values);
    free(c_prior);
    
    if (result != 0) {
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    viterbi_free(&_solver);
}

- (NSUInteger)stateSpaceSize {
    return _solver.state_space_size;
}

- (double)currentTime {
    return _solver.current_time;
}

- (double)mapProbability {
    return _solver.map_probability;
}

- (BOOL)stepWithObservation:(double)observation error:(NSError**)error {
    int result = viterbi_step(&_solver, observation);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Viterbi step failed"}];
    }
    return result == 0;
}

- (BOOL)getMAPEstimate:(double*)mapEstimate
         mapProbability:(double*)mapProbability
                 error:(NSError**)error {
    if (!mapEstimate) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    int result = viterbi_get_map(&_solver, mapEstimate, mapProbability);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"MAP estimation failed"}];
    }
    return result == 0;
}

@end

// ============================================================================
// Real-Time Bayesian Solver
// ============================================================================

@implementation DDRKAMRealTimeBayesianSolver {
    RealTimeBayesianSolver _solver;
}

- (instancetype)initWithMode:(DDRKAMBayesianMode)mode
              stateSpaceSize:(NSUInteger)stateSpaceSize
                 stateValues:(NSArray<NSNumber*>*)stateValues
            transitionMatrix:(NSArray<NSArray<NSNumber*>*>*)transitionMatrix
                       prior:(NSArray<NSNumber*>*)prior
    observationNoiseVariance:(double)noiseVariance
                       error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    // Convert to C arrays
    double* c_state_values = (double*)malloc(stateSpaceSize * sizeof(double));
    double** c_transition = (double**)malloc(stateSpaceSize * sizeof(double*));
    double* c_prior = (double*)malloc(stateSpaceSize * sizeof(double));
    
    if (!c_state_values || !c_transition || !c_prior) {
        free(c_state_values);
        free(c_transition);
        free(c_prior);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return nil;
    }
    
    for (NSUInteger i = 0; i < stateSpaceSize; i++) {
        c_state_values[i] = [stateValues[i] doubleValue];
        c_prior[i] = [prior[i] doubleValue];
        
        c_transition[i] = (double*)malloc(stateSpaceSize * sizeof(double));
        NSArray<NSNumber*>* row = transitionMatrix[i];
        for (NSUInteger j = 0; j < stateSpaceSize; j++) {
            c_transition[i][j] = [row[j] doubleValue];
        }
    }
    
    BayesianMode c_mode = (BayesianMode)mode;
    int result = realtime_bayesian_init(&_solver, c_mode, stateSpaceSize,
                                       c_state_values, c_transition, c_prior, noiseVariance);
    
    for (NSUInteger i = 0; i < stateSpaceSize; i++) {
        free(c_transition[i]);
    }
    free(c_transition);
    free(c_state_values);
    free(c_prior);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    realtime_bayesian_free(&_solver);
}

- (DDRKAMBayesianMode)mode {
    return (DDRKAMBayesianMode)_solver.mode;
}

- (NSUInteger)stateSpaceSize {
    return _solver.state_space_size;
}

- (double)currentTime {
    return _solver.current_time;
}

- (BOOL)stepAtTime:(double)t
      observation:(double)observation
            yOut:(double*)yOut
            error:(NSError**)error {
    if (!yOut) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    int result = realtime_bayesian_step(&_solver, t, observation, yOut);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Step failed"}];
    }
    return result == 0;
}

- (BOOL)getStatisticsMean:(double*)mean
                 variance:(double*)variance
                     yMAP:(double*)yMAP
                    error:(NSError**)error {
    if (!mean || !variance) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    int result = realtime_bayesian_get_statistics(&_solver, mean, variance, yMAP);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Statistics computation failed"}];
    }
    return result == 0;
}

@end

// ============================================================================
// Randomized DP Solver (simplified wrapper - full implementation would require C function pointers)
// ============================================================================

@implementation DDRKAMRandomizedDPSolver {
    RandomizedDPSolver _solver;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                           numSamples:(NSUInteger)numSamples
                         numControls:(NSUInteger)numControls
                   controlCandidates:(NSArray<NSNumber*>*)controlCandidates
                             odeFunc:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                           odeParams:(void* _Nullable)odeParams
                       costFunction:(double(*)(double t, const double* y, double u, void* params))costFunc
                        costParams:(void* _Nullable)costParams
                    samplingRadius:(double)radius
                       ucbConstant:(double)ucbConstant
                             error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    if (controlCandidates.count != numControls) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid control candidates count"}];
        }
        return nil;
    }
    
    double* c_controls = (double*)malloc(numControls * sizeof(double));
    if (!c_controls) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return nil;
    }
    
    for (NSUInteger i = 0; i < numControls; i++) {
        c_controls[i] = [controlCandidates[i] doubleValue];
    }
    
    int result = randomized_dp_init(&_solver, stateDim, numSamples, numControls,
                                   c_controls, odeFunc, odeParams,
                                   costFunc, costParams, radius, ucbConstant);
    
    free(c_controls);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    randomized_dp_free(&_solver);
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (NSUInteger)numSamples {
    return _solver.num_samples;
}

- (NSUInteger)numControls {
    return _solver.num_controls;
}

- (double)samplingRadius {
    return _solver.sampling_radius;
}

- (double)ucbConstant {
    return _solver.ucb_constant;
}

- (uint64_t)stepCount {
    return _solver.step_count;
}

- (BOOL)stepAtTime:(double)t
      currentState:(NSArray<NSNumber*>*)yCurrent
           nextState:(NSArray<NSNumber*>**)yNext
    optimalControl:(double*)optimalControl
             error:(NSError**)error {
    if (!yCurrent || !yNext || !optimalControl) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    if (yCurrent.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"State dimension mismatch"}];
        }
        return NO;
    }
    
    double* c_y_current = (double*)malloc(_solver.state_dim * sizeof(double));
    double* c_y_next = (double*)malloc(_solver.state_dim * sizeof(double));
    
    if (!c_y_current || !c_y_next) {
        free(c_y_current);
        free(c_y_next);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y_current[i] = [yCurrent[i] doubleValue];
    }
    
    int result = randomized_dp_step(&_solver, t, c_y_current, c_y_next, optimalControl);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* nextState = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            [nextState addObject:@(c_y_next[i])];
        }
        *yNext = [nextState copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Step failed"}];
    }
    
    free(c_y_current);
    free(c_y_next);
    
    return result == 0;
}

- (BOOL)getValueAtTime:(double)t
                 state:(NSArray<NSNumber*>*)y
      valueEstimate:(double*)valueEstimate
      valueVariance:(double* _Nullable)valueVariance
               error:(NSError**)error {
    if (!y || !valueEstimate) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y[i] = [y[i] doubleValue];
    }
    
    int result = randomized_dp_get_value(&_solver, t, c_y, valueEstimate, valueVariance);
    
    free(c_y);
    
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMBayesianSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Value estimation failed"}];
    }
    
    return result == 0;
}

@end
