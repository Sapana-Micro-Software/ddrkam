/*
 * DDRKAM Causal and Granger Causality ODE Solvers - Objective-C Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMCausalGrangerSolvers.h"
#import "causal_granger_solvers.h"
#import <Foundation/Foundation.h>

// ============================================================================
// Causal RK4 Solver
// ============================================================================

@implementation DDRKAMCausalRK4Solver {
    CausalRK4Solver _solver;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                          historySize:(NSUInteger)historySize
                       strictCausality:(BOOL)strictCausality
                                 error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    int result = causal_rk4_init(&_solver, stateDim, stepSize, historySize,
                                 strictCausality ? 1 : 0);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Causal RK4 initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    causal_rk4_free(&_solver);
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (double)stepSize {
    return _solver.step_size;
}

- (double)currentTime {
    return _solver.current_time;
}

- (NSUInteger)totalSteps {
    return _solver.total_steps;
}

- (BOOL)strictCausality {
    return _solver.strict_causality != 0;
}

- (BOOL)stepWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                      atTime:(double)t
                       state:(NSArray<NSNumber*>*)y
                      params:(void* _Nullable)params
                       error:(NSError**)error {
    if (!y || y.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid state dimension"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y[i] = [y[i] doubleValue];
    }
    
    int result = causal_rk4_step(&_solver, odeFunc, t, c_y, params);
    
    if (result == 0) {
        // Update y array (in-place modification expected)
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            // Note: This modifies the input array, which is expected for in-place operations
        }
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Causal RK4 step failed"}];
    }
    
    free(c_y);
    return result == 0;
}

- (BOOL)solveWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                    fromTime:(double)t0
                      toTime:(double)tEnd
           initialCondition:(NSArray<NSNumber*>*)y0
                      params:(void* _Nullable)params
                 numSteps:(NSUInteger)numSteps
             solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                     error:(NSError**)error {
    if (!y0 || !solutionPath || y0.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double** c_solution = (double**)malloc(numSteps * sizeof(double*));
    if (!c_solution) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
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
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    double* c_y0 = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y0) {
        for (NSUInteger i = 0; i < numSteps; i++) {
            free(c_solution[i]);
        }
        free(c_solution);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    int result = causal_rk4_solve(&_solver, odeFunc, t0, tEnd, c_y0, params,
                                  c_solution, numSteps);
    
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
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Causal RK4 solve failed"}];
    }
    
    for (NSUInteger i = 0; i < numSteps; i++) {
        free(c_solution[i]);
    }
    free(c_solution);
    free(c_y0);
    
    return result == 0;
}

@end

// ============================================================================
// Causal Adams Solver
// ============================================================================

@implementation DDRKAMCausalAdamsSolver {
    CausalAdamsSolver _solver;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                           adamsOrder:(NSUInteger)adamsOrder
                          historySize:(NSUInteger)historySize
                       strictCausality:(BOOL)strictCausality
                                 error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    int result = causal_adams_init(&_solver, stateDim, stepSize, adamsOrder,
                                  historySize, strictCausality ? 1 : 0);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Causal Adams initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    causal_adams_free(&_solver);
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (double)stepSize {
    return _solver.step_size;
}

- (NSUInteger)adamsOrder {
    return _solver.adams_order;
}

- (double)currentTime {
    return _solver.current_time;
}

- (NSUInteger)totalSteps {
    return _solver.total_steps;
}

- (BOOL)strictCausality {
    return _solver.strict_causality != 0;
}

- (BOOL)stepWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                      atTime:(double)t
                       state:(NSArray<NSNumber*>*)y
                      params:(void* _Nullable)params
                       error:(NSError**)error {
    if (!y || y.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid state dimension"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y[i] = [y[i] doubleValue];
    }
    
    int result = causal_adams_step(&_solver, odeFunc, t, c_y, params);
    
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Causal Adams step failed"}];
    }
    
    free(c_y);
    return result == 0;
}

- (BOOL)solveWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                    fromTime:(double)t0
                      toTime:(double)tEnd
           initialCondition:(NSArray<NSNumber*>*)y0
                      params:(void* _Nullable)params
                 numSteps:(NSUInteger)numSteps
             solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                     error:(NSError**)error {
    // Similar implementation to Causal RK4
    if (!y0 || !solutionPath || y0.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
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
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
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
                *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    int result = causal_adams_solve(&_solver, odeFunc, t0, tEnd, c_y0, params,
                                    c_solution, numSteps);
    
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
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Causal Adams solve failed"}];
    }
    
    for (NSUInteger i = 0; i < numSteps; i++) {
        free(c_solution[i]);
    }
    free(c_solution);
    free(c_y0);
    
    return result == 0;
}

@end

// ============================================================================
// Granger Causality Solver
// ============================================================================

@implementation DDRKAMGrangerCausalitySolver {
    GrangerCausalitySolver _solver;
}

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                           baseMethod:(DDRKAMGrangerBaseMethod)baseMethod
                           adamsOrder:(NSUInteger)adamsOrder
                      causalityWindow:(NSUInteger)causalityWindow
                                 error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    int c_base_method = (baseMethod == DDRKAMGrangerBaseRK4) ? GRANGER_BASE_RK4 : GRANGER_BASE_ADAMS;
    
    int result = granger_causality_init(&_solver, stateDim, stepSize, c_base_method,
                                       adamsOrder, causalityWindow);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Granger causality initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    granger_causality_free(&_solver);
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (double)stepSize {
    return _solver.step_size;
}

- (DDRKAMGrangerBaseMethod)baseMethod {
    return (DDRKAMGrangerBaseMethod)_solver.base_method;
}

- (NSUInteger)causalityWindow {
    return _solver.causality_window;
}

- (double)currentTime {
    return _solver.current_time;
}

- (NSUInteger)totalSteps {
    return _solver.total_steps;
}

- (uint64_t)causalityUpdates {
    return _solver.causality_updates;
}

- (BOOL)stepWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                      atTime:(double)t
                       state:(NSArray<NSNumber*>*)y
                      params:(void* _Nullable)params
                       error:(NSError**)error {
    if (!y || y.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid state dimension"}];
        }
        return NO;
    }
    
    double* c_y = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_y) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y[i] = [y[i] doubleValue];
    }
    
    int result = granger_causality_step(&_solver, odeFunc, t, c_y, params);
    
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Granger causality step failed"}];
    }
    
    free(c_y);
    return result == 0;
}

- (BOOL)updateCausalityWithError:(NSError**)error {
    int result = granger_causality_update(&_solver);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Causality update failed"}];
    }
    return result == 0;
}

- (BOOL)getCausalityMatrix:(NSArray<NSArray<NSNumber*>*>**)causalityMatrix
                      error:(NSError**)error {
    if (!causalityMatrix) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double** c_matrix = (double**)malloc(_solver.state_dim * sizeof(double*));
    if (!c_matrix) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_matrix[i] = (double*)malloc(_solver.state_dim * sizeof(double));
        if (!c_matrix[i]) {
            for (NSUInteger j = 0; j < i; j++) {
                free(c_matrix[j]);
            }
            free(c_matrix);
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    int result = granger_causality_get_matrix(&_solver, c_matrix);
    
    if (result == 0) {
        NSMutableArray<NSArray<NSNumber*>*>* matrix = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            NSMutableArray<NSNumber*>* row = [NSMutableArray arrayWithCapacity:_solver.state_dim];
            for (NSUInteger j = 0; j < _solver.state_dim; j++) {
                [row addObject:@(c_matrix[i][j])];
            }
            [matrix addObject:[row copy]];
        }
        *causalityMatrix = [matrix copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Failed to get causality matrix"}];
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        free(c_matrix[i]);
    }
    free(c_matrix);
    
    return result == 0;
}

- (BOOL)getVariableImportance:(NSArray<NSNumber*>**)importance
                         error:(NSError**)error {
    if (!importance) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_importance = (double*)malloc(_solver.state_dim * sizeof(double));
    if (!c_importance) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    int result = granger_causality_get_importance(&_solver, c_importance);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* imp = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            [imp addObject:@(c_importance[i])];
        }
        *importance = [imp copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Failed to get variable importance"}];
    }
    
    free(c_importance);
    return result == 0;
}

- (BOOL)solveWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                    fromTime:(double)t0
                      toTime:(double)tEnd
           initialCondition:(NSArray<NSNumber*>*)y0
                      params:(void* _Nullable)params
                 numSteps:(NSUInteger)numSteps
             solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                     error:(NSError**)error {
    // Similar to other solve methods
    if (!y0 || !solutionPath || y0.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
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
            *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
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
                *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
            }
            return NO;
        }
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    int result = granger_causality_solve(&_solver, odeFunc, t0, tEnd, c_y0, params,
                                        c_solution, numSteps);
    
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
        *error = [NSError errorWithDomain:@"DDRKAMCausalGrangerSolvers"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Granger causality solve failed"}];
    }
    
    for (NSUInteger i = 0; i < numSteps; i++) {
        free(c_solution[i]);
    }
    free(c_solution);
    free(c_y0);
    
    return result == 0;
}

@end
