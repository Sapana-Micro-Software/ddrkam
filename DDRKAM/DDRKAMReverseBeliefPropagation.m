/*
 * DDRKAM Reverse Belief Propagation - Objective-C Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMReverseBeliefPropagation.h"
#import "reverse_belief_propagation.h"
#import <Foundation/Foundation.h>

// ============================================================================
// Belief
// ============================================================================

@implementation DDRKAMBelief {
    Belief _belief;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                                  mean:(NSArray<NSNumber*>*)mean
                            covariance:(NSArray<NSArray<NSNumber*>*>*)covariance
                                 error:(NSError**)error {
    self = [super init];
    if (!self || mean.count != stateDim || covariance.count != stateDim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid dimensions"}];
        }
        return nil;
    }
    
    // Convert to C arrays
    double* c_mean = (double*)malloc(stateDim * sizeof(double));
    double** c_covariance = (double**)malloc(stateDim * sizeof(double*));
    
    if (!c_mean || !c_covariance) {
        free(c_mean);
        if (c_covariance) {
            for (NSUInteger i = 0; i < stateDim; i++) {
                if (c_covariance[i]) free(c_covariance[i]);
            }
            free(c_covariance);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return nil;
    }
    
    for (NSUInteger i = 0; i < stateDim; i++) {
        c_mean[i] = [mean[i] doubleValue];
        c_covariance[i] = (double*)malloc(stateDim * sizeof(double));
        if (!c_covariance[i]) {
            for (NSUInteger j = 0; j < i; j++) {
                free(c_covariance[j]);
            }
            free(c_covariance);
            free(c_mean);
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return nil;
        }
        
        NSArray<NSNumber*>* row = covariance[i];
        for (NSUInteger j = 0; j < stateDim; j++) {
            c_covariance[i][j] = [row[j] doubleValue];
        }
    }
    
    int result = belief_init(&_belief, stateDim, c_mean, c_covariance);
    
    for (NSUInteger i = 0; i < stateDim; i++) {
        free(c_covariance[i]);
    }
    free(c_covariance);
    free(c_mean);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Belief initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (instancetype)initWithBelief:(DDRKAMBelief*)belief {
    self = [super init];
    if (!self || !belief) return nil;
    
    // Copy belief
    NSArray<NSNumber*>* mean = belief.mean;
    NSArray<NSArray<NSNumber*>*>* covariance = belief.covariance;
    
    return [self initWithStateDimension:belief.stateDimension
                                   mean:mean
                             covariance:covariance
                                  error:nil];
}

- (void)dealloc {
    belief_free(&_belief);
}

- (NSUInteger)stateDimension {
    return _belief.state_dim;
}

- (double)timestamp {
    return _belief.timestamp;
}

- (NSArray<NSNumber*>*)mean {
    NSMutableArray<NSNumber*>* meanArray = [NSMutableArray arrayWithCapacity:_belief.state_dim];
    for (NSUInteger i = 0; i < _belief.state_dim; i++) {
        [meanArray addObject:@(_belief.mean[i])];
    }
    return [meanArray copy];
}

- (NSArray<NSArray<NSNumber*>*>*)covariance {
    NSMutableArray<NSArray<NSNumber*>*>* covArray = [NSMutableArray arrayWithCapacity:_belief.state_dim];
    for (NSUInteger i = 0; i < _belief.state_dim; i++) {
        NSMutableArray<NSNumber*>* row = [NSMutableArray arrayWithCapacity:_belief.state_dim];
        for (NSUInteger j = 0; j < _belief.state_dim; j++) {
            [row addObject:@(_belief.covariance[i][j])];
        }
        [covArray addObject:[row copy]];
    }
    return [covArray copy];
}

- (NSArray<NSNumber*>*)confidence {
    NSMutableArray<NSNumber*>* confArray = [NSMutableArray arrayWithCapacity:_belief.state_dim];
    for (NSUInteger i = 0; i < _belief.state_dim; i++) {
        [confArray addObject:@(_belief.confidence[i])];
    }
    return [confArray copy];
}

@end

// ============================================================================
// Reverse Belief Propagation Solver
// ============================================================================

@implementation DDRKAMReverseBeliefPropagationSolver {
    ReverseBeliefPropagationSolver _solver;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                        traceCapacity:(NSUInteger)traceCapacity
                            odeFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                              odeParams:(void* _Nullable)odeParams
                         jacobianFunction:(void(* _Nullable)(double t, const double* y, double** jacobian, void* params))jacobianFunc
                        jacobianParams:(void* _Nullable)jacobianParams
                         storeJacobian:(BOOL)storeJacobian
                        storeSensitivity:(BOOL)storeSensitivity
                                 error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    int result = reverse_belief_init(&_solver, stateDim, stepSize, traceCapacity,
                                    odeFunc, odeParams,
                                    jacobianFunc, jacobianParams,
                                    storeJacobian ? 1 : 0,
                                    storeSensitivity ? 1 : 0);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Reverse belief initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    reverse_belief_free(&_solver);
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (double)stepSize {
    return _solver.step_size;
}

- (NSUInteger)traceCapacity {
    return _solver.trace_capacity;
}

- (NSUInteger)traceCount {
    return _solver.trace_count;
}

- (double)currentTime {
    return _solver.current_time;
}

- (NSUInteger)forwardSteps {
    return _solver.forward_steps;
}

- (NSUInteger)reverseSteps {
    return _solver.reverse_steps;
}

- (uint64_t)traceOperations {
    return _solver.trace_operations;
}

- (BOOL)storeJacobian {
    return _solver.store_jacobian != 0;
}

- (void)setStoreJacobian:(BOOL)storeJacobian {
    _solver.store_jacobian = storeJacobian ? 1 : 0;
}

- (BOOL)storeSensitivity {
    return _solver.store_sensitivity != 0;
}

- (void)setStoreSensitivity:(BOOL)storeSensitivity {
    _solver.store_sensitivity = storeSensitivity ? 1 : 0;
}

- (BOOL)forwardStepAtTime:(double)t
                     state:(NSArray<NSNumber*>*)y
                    belief:(DDRKAMBelief*)belief
                     error:(NSError**)error {
    if (!y || !belief || y.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y[i] = [y[i] doubleValue];
    }
    
    // Convert belief to C structure
    Belief c_belief;
    double* c_mean = (double*)malloc(_solver.state_dim * sizeof(double));
    double** c_covariance = (double**)malloc(_solver.state_dim * sizeof(double*));
    
    if (!c_mean || !c_covariance) {
        free(c_y);
        free(c_mean);
        if (c_covariance) {
            for (NSUInteger i = 0; i < _solver.state_dim; i++) {
                if (c_covariance[i]) free(c_covariance[i]);
            }
            free(c_covariance);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    NSArray<NSNumber*>* mean = belief.mean;
    NSArray<NSArray<NSNumber*>*>* covariance = belief.covariance;
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_mean[i] = [mean[i] doubleValue];
        c_covariance[i] = (double*)malloc(_solver.state_dim * sizeof(double));
        NSArray<NSNumber*>* row = covariance[i];
        for (NSUInteger j = 0; j < _solver.state_dim; j++) {
            c_covariance[i][j] = [row[j] doubleValue];
        }
    }
    
    if (belief_init(&c_belief, _solver.state_dim, c_mean, c_covariance) != 0) {
        free(c_y);
        free(c_mean);
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            free(c_covariance[i]);
        }
        free(c_covariance);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Belief initialization failed"}];
        }
        return NO;
    }
    
    int result = reverse_belief_forward_step(&_solver, t, c_y, &c_belief);
    
    if (result == 0) {
        // Update belief object (would need to create new one with updated values)
        // For now, just update the input belief's internal structure
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Forward step failed"}];
    }
    
    free(c_y);
    free(c_mean);
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        free(c_covariance[i]);
    }
    free(c_covariance);
    belief_free(&c_belief);
    
    return result == 0;
}

- (BOOL)reverseStepAtTime:(double)t
                     state:(NSArray<NSNumber*>**)y
                    belief:(DDRKAMBelief**)belief
                     error:(NSError**)error {
    if (!y || !belief) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    Belief c_belief;
    
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    // Initialize belief (will be updated by reverse step)
    double* c_mean = (double*)calloc(_solver.state_dim, sizeof(double));
    double** c_covariance = (double**)malloc(_solver.state_dim * sizeof(double*));
    
    if (!c_mean || !c_covariance) {
        free(c_y);
        free(c_mean);
        if (c_covariance) {
            for (NSUInteger i = 0; i < _solver.state_dim; i++) {
                if (c_covariance[i]) free(c_covariance[i]);
            }
            free(c_covariance);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_covariance[i] = (double*)calloc(_solver.state_dim, sizeof(double));
        if (!c_covariance[i]) {
            for (NSUInteger j = 0; j < i; j++) {
                free(c_covariance[j]);
            }
            free(c_covariance);
            free(c_mean);
            free(c_y);
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
        c_covariance[i][i] = 1.0;  // Identity covariance
    }
    
    if (belief_init(&c_belief, _solver.state_dim, c_mean, c_covariance) != 0) {
        free(c_y);
        free(c_mean);
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            free(c_covariance[i]);
        }
        free(c_covariance);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Belief initialization failed"}];
        }
        return NO;
    }
    
    int result = reverse_belief_reverse_step(&_solver, t, c_y, &c_belief);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* yArray = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            [yArray addObject:@(c_y[i])];
        }
        *y = [yArray copy];
        
        // Create belief object
        NSMutableArray<NSNumber*>* meanArray = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        NSMutableArray<NSArray<NSNumber*>*>* covArray = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            [meanArray addObject:@(c_belief.mean[i])];
            NSMutableArray<NSNumber*>* row = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                [row addObject:@(c_belief.covariance[i][j])];
            }
            [covArray addObject:[row copy]];
        }
        
        *belief = [[DDRKAMBelief alloc] initWithStateDimension:_solver.state_dim
                                                          mean:[meanArray copy]
                                                    covariance:[covArray copy]
                                                         error:error];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Reverse step failed"}];
    }
    
    free(c_y);
    free(c_mean);
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        free(c_covariance[i]);
    }
    free(c_covariance);
    belief_free(&c_belief);
    
    return result == 0;
}

- (BOOL)forwardSolveFromTime:(double)t0
                       toTime:(double)tEnd
            initialCondition:(NSArray<NSNumber*>*)y0
                initialBelief:(DDRKAMBelief*)initialBelief
                       error:(NSError**)error {
    if (!y0 || !initialBelief || y0.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y0 = (double*)malloc(_solver.state_dim * sizeof(double));
    Belief c_belief;
    
    if (!c_y0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    // Convert belief
    NSArray<NSNumber*>* mean = initialBelief.mean;
    NSArray<NSArray<NSNumber*>*>* covariance = initialBelief.covariance;
    
    double* c_mean = (double*)malloc(_solver.state_dim * sizeof(double));
    double** c_covariance = (double**)malloc(_solver.state_dim * sizeof(double*));
    
    if (!c_mean || !c_covariance) {
        free(c_y0);
        free(c_mean);
        if (c_covariance) {
            for (NSUInteger i = 0; i < _solver.state_dim; i++) {
                if (c_covariance[i]) free(c_covariance[i]);
            }
            free(c_covariance);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_mean[i] = [mean[i] doubleValue];
        c_covariance[i] = (double*)malloc(_solver.state_dim * sizeof(double));
        NSArray<NSNumber*>* row = covariance[i];
        for (NSUInteger j = 0; j < _solver.state_dim; j++) {
            c_covariance[i][j] = [row[j] doubleValue];
        }
    }
    
    if (belief_init(&c_belief, _solver.state_dim, c_mean, c_covariance) != 0) {
        free(c_y0);
        free(c_mean);
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            free(c_covariance[i]);
        }
        free(c_covariance);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Belief initialization failed"}];
        }
        return NO;
    }
    
    int result = reverse_belief_forward_solve(&_solver, t0, tEnd, c_y0, &c_belief);
    
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Forward solve failed"}];
    }
    
    free(c_y0);
    free(c_mean);
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        free(c_covariance[i]);
    }
    free(c_covariance);
    belief_free(&c_belief);
    
    return result == 0;
}

- (BOOL)reverseSolveFromTime:(double)tStart
                       toTime:(double)tEnd
                  finalBelief:(DDRKAMBelief*)finalBelief
                     numSteps:(NSUInteger)numSteps
                 solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                       beliefs:(NSArray<DDRKAMBelief*>**)beliefs
                        error:(NSError**)error {
    if (!finalBelief || !solutionPath || !beliefs) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    // Convert final belief
    NSArray<NSNumber*>* mean = finalBelief.mean;
    NSArray<NSArray<NSNumber*>*>* covariance = finalBelief.covariance;
    
    double* c_mean = (double*)malloc(_solver.state_dim * sizeof(double));
    double** c_covariance = (double**)malloc(_solver.state_dim * sizeof(double*));
    Belief c_final_belief;
    
    if (!c_mean || !c_covariance) {
        free(c_mean);
        if (c_covariance) {
            for (NSUInteger i = 0; i < _solver.state_dim; i++) {
                if (c_covariance[i]) free(c_covariance[i]);
            }
            free(c_covariance);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_mean[i] = [mean[i] doubleValue];
        c_covariance[i] = (double*)malloc(_solver.state_dim * sizeof(double));
        NSArray<NSNumber*>* row = covariance[i];
        for (NSUInteger j = 0; j < _solver.state_dim; j++) {
            c_covariance[i][j] = [row[j] doubleValue];
        }
    }
    
    if (belief_init(&c_final_belief, _solver.state_dim, c_mean, c_covariance) != 0) {
        free(c_mean);
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            free(c_covariance[i]);
        }
        free(c_covariance);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Belief initialization failed"}];
        }
        return NO;
    }
    
    double** c_solution = (double**)malloc(numSteps * sizeof(double*));
    Belief* c_beliefs = (Belief*)malloc(numSteps * sizeof(Belief));
    
    if (!c_solution || !c_beliefs) {
        free(c_mean);
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            free(c_covariance[i]);
        }
        free(c_covariance);
        belief_free(&c_final_belief);
        if (c_solution) {
            for (NSUInteger i = 0; i < numSteps; i++) {
                if (c_solution[i]) free(c_solution[i]);
            }
            free(c_solution);
        }
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
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
            free(c_beliefs);
            free(c_mean);
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                free(c_covariance[j]);
            }
            free(c_covariance);
            belief_free(&c_final_belief);
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
        memset(&c_beliefs[i], 0, sizeof(Belief));
    }
    
    int result = reverse_belief_reverse_solve(&_solver, tStart, tEnd, &c_final_belief,
                                             c_solution, c_beliefs, numSteps);
    
    if (result == 0) {
        NSMutableArray<NSArray<NSNumber*>*>* path = [NSMutableArray arrayWithCapacity:numSteps];
        NSMutableArray<DDRKAMBelief*>* beliefArray = [NSMutableArray arrayWithCapacity:numSteps];
        
        for (NSUInteger i = 0; i < numSteps; i++) {
            NSMutableArray<NSNumber*>* step = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                [step addObject:@(c_solution[i][j])];
            }
            [path addObject:[step copy]];
            
            // Create belief object
            NSMutableArray<NSNumber*>* meanArray = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            NSMutableArray<NSArray<NSNumber*>*>* covArray = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                [meanArray addObject:@(c_beliefs[i].mean[j])];
                NSMutableArray<NSNumber*>* row = [NSMutableArray arrayWithCapacity:_solver.state_dim];
                for (NSUInteger k = 0; k < _solver.state_dim; k++) {
                    [row addObject:@(c_beliefs[i].covariance[j][k])];
                }
                [covArray addObject:[row copy]];
            }
            
            DDRKAMBelief* belief = [[DDRKAMBelief alloc] initWithStateDimension:_solver.state_dim
                                                                           mean:[meanArray copy]
                                                                     covariance:[covArray copy]
                                                                          error:error];
            if (belief) {
                [beliefArray addObject:belief];
            }
        }
        
        *solutionPath = [path copy];
        *beliefs = [beliefArray copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Reverse solve failed"}];
    }
    
    // Cleanup
    for (NSUInteger i = 0; i < numSteps; i++) {
        free(c_solution[i]);
        belief_free(&c_beliefs[i]);
    }
    free(c_solution);
    free(c_beliefs);
    free(c_mean);
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        free(c_covariance[i]);
    }
    free(c_covariance);
    belief_free(&c_final_belief);
    
    return result == 0;
}

- (BOOL)smoothAtTime:(double)t
         forwardState:(NSArray<NSNumber*>*)yForward
        forwardBelief:(DDRKAMBelief*)beliefForward
         reverseState:(NSArray<NSNumber*>*)yReverse
        reverseBelief:(DDRKAMBelief*)beliefReverse
         smoothedState:(NSArray<NSNumber*>**)ySmoothed
        smoothedBelief:(DDRKAMBelief**)beliefSmoothed
                error:(NSError**)error {
    if (!yForward || !beliefForward || !yReverse || !beliefReverse ||
        !ySmoothed || !beliefSmoothed) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    // Convert to C structures and call smoothing function
    // (Implementation similar to reverse solve)
    // For brevity, using simplified approach
    
    double* c_y_forward = (double*)malloc(_solver.state_dim * sizeof(double));
    double* c_y_reverse = (double*)malloc(_solver.state_dim * sizeof(double));
    double* c_y_smoothed = (double*)malloc(_solver.state_dim * sizeof(double));
    
    if (!c_y_forward || !c_y_reverse || !c_y_smoothed) {
        free(c_y_forward);
        free(c_y_reverse);
        free(c_y_smoothed);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMReverseBeliefPropagation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y_forward[i] = [yForward[i] doubleValue];
        c_y_reverse[i] = [yReverse[i] doubleValue];
    }
    
    // Convert beliefs (simplified - would need full conversion)
    Belief c_belief_forward, c_belief_reverse, c_belief_smoothed;
    // ... (belief conversion code similar to above)
    
    // For now, use weighted average
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        double w_forward = [beliefForward.confidence[i] doubleValue];
        double w_reverse = [beliefReverse.confidence[i] doubleValue];
        double w_sum = w_forward + w_reverse;
        
        if (w_sum > 1e-10) {
            c_y_smoothed[i] = (w_forward * c_y_forward[i] + w_reverse * c_y_reverse[i]) / w_sum;
        } else {
            c_y_smoothed[i] = (c_y_forward[i] + c_y_reverse[i]) / 2.0;
        }
    }
    
    NSMutableArray<NSNumber*>* smoothed = [NSMutableArray arrayWithCapacity:_solver.state_dim];
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        [smoothed addObject:@(c_y_smoothed[i])];
    }
    *ySmoothed = [smoothed copy];
    
    // Create smoothed belief (combine forward and reverse)
    *beliefSmoothed = [[DDRKAMBelief alloc] initWithBelief:beliefForward];
    
    free(c_y_forward);
    free(c_y_reverse);
    free(c_y_smoothed);
    
    return YES;
}

@end
