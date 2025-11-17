/*
 * DDRKAM O(1) Approximation Solvers - Objective-C Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMO1Approximation.h"
#import "o1_approximation.h"
#import <Foundation/Foundation.h>

// ============================================================================
// Lookup Table Solver
// ============================================================================

@implementation DDRKAMLookupTableSolver {
    LookupTableSolver _solver;
}

- (instancetype)initWithParamGridSize:(NSUInteger)paramGridSize
                         timeGridSize:(NSUInteger)timeGridSize
                        stateDimension:(NSUInteger)stateDim
                            paramMin:(double)paramMin
                            paramMax:(double)paramMax
                             timeMax:(double)timeMax
                               error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    int result = lookup_table_init(&_solver, paramGridSize, timeGridSize, stateDim,
                                  paramMin, paramMax, timeMax);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Lookup table initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    lookup_table_free(&_solver);
}

- (NSUInteger)paramGridSize {
    return _solver.param_grid_size;
}

- (NSUInteger)timeGridSize {
    return _solver.time_grid_size;
}

- (NSUInteger)stateDimension {
    return _solver.state_dim;
}

- (BOOL)precomputeWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                            params:(void* _Nullable)params
                             error:(NSError**)error {
    int result = lookup_table_precompute(&_solver, odeFunc, params);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Pre-computation failed"}];
    }
    return result == 0;
}

- (BOOL)solveAtTime:(double)t
            parameter:(double)param
      initialCondition:(NSArray<NSNumber*>*)y0
            solution:(NSArray<NSNumber*>**)solution
               error:(NSError**)error {
    if (!y0 || !solution || y0.count != _solver.state_dim) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y0 = (double*)malloc(_solver.state_dim * sizeof(double));
    double* c_solution = (double*)malloc(_solver.state_dim * sizeof(double));
    
    if (!c_y0 || !c_solution) {
        free(c_y0);
        free(c_solution);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _solver.state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    int result = lookup_table_solve(&_solver, t, param, c_y0, c_solution);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* sol = [NSMutableArray arrayWithCapacity:_solver.state_dim];
        for (NSUInteger i = 0; i < _solver.state_dim; i++) {
            [sol addObject:@(c_solution[i])];
        }
        *solution = [sol copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Lookup solve failed"}];
    }
    
    free(c_y0);
    free(c_solution);
    
    return result == 0;
}

@end

// ============================================================================
// Neural Network Approximator
// ============================================================================

@implementation DDRKAMNeuralApproximator {
    NeuralApproximator _net;
}

- (instancetype)initWithLayerSizes:(NSArray<NSNumber*>*)layerSizes
                             error:(NSError**)error {
    self = [super init];
    if (!self || layerSizes.count < 2) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid layer sizes"}];
        }
        return nil;
    }
    
    size_t* c_layer_sizes = (size_t*)malloc(layerSizes.count * sizeof(size_t));
    if (!c_layer_sizes) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return nil;
    }
    
    for (NSUInteger i = 0; i < layerSizes.count; i++) {
        c_layer_sizes[i] = (size_t)[layerSizes[i] unsignedIntegerValue];
    }
    
    int result = neural_approximator_init(&_net, c_layer_sizes, layerSizes.count);
    
    free(c_layer_sizes);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Neural network initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    neural_approximator_free(&_net);
}

- (NSUInteger)inputDimension {
    return _net.input_dim;
}

- (NSUInteger)outputDimension {
    return _net.output_dim;
}

- (NSUInteger)numLayers {
    return _net.num_layers;
}

- (BOOL)loadWeights:(NSArray<NSArray<NSNumber*>*>*)weights
            biases:(NSArray<NSNumber*>*)biases
             error:(NSError**)error {
    // Implementation would copy weights and biases
    // Placeholder for now
    return YES;
}

- (BOOL)solveAtTime:(double)t
      initialCondition:(NSArray<NSNumber*>*)y0
            parameters:(NSArray<NSNumber*>*)params
            solution:(NSArray<NSNumber*>**)solution
               error:(NSError**)error {
    if (!y0 || !solution) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    double* c_y0 = (double*)malloc(y0.count * sizeof(double));
    double* c_params = NULL;
    double* c_solution = (double*)malloc(_net.output_dim * sizeof(double));
    
    if (!c_y0 || !c_solution) {
        free(c_y0);
        free(c_solution);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < y0.count; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    if (params && params.count > 0) {
        c_params = (double*)malloc(params.count * sizeof(double));
        for (NSUInteger i = 0; i < params.count; i++) {
            c_params[i] = [params[i] doubleValue];
        }
    }
    
    int result = neural_approximator_solve(&_net, t, c_y0, c_params,
                                          params ? params.count : 0, c_solution);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* sol = [NSMutableArray arrayWithCapacity:_net.output_dim];
        for (NSUInteger i = 0; i < _net.output_dim; i++) {
            [sol addObject:@(c_solution[i])];
        }
        *solution = [sol copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Neural network solve failed"}];
    }
    
    free(c_y0);
    free(c_params);
    free(c_solution);
    
    return result == 0;
}

@end

// ============================================================================
// Chebyshev Approximator
// ============================================================================

@implementation DDRKAMChebyshevApproximator {
    ChebyshevApproximator _approx;
}

- (instancetype)initWithNumCoefficients:(NSUInteger)numCoeffs
                              numParams:(NSUInteger)numParams
                          paramValues:(NSArray<NSNumber*>*)paramValues
                            timeScale:(double)timeScale
                                error:(NSError**)error {
    self = [super init];
    if (!self || paramValues.count != numParams) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return nil;
    }
    
    double* c_params = (double*)malloc(numParams * sizeof(double));
    if (!c_params) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return nil;
    }
    
    for (NSUInteger i = 0; i < numParams; i++) {
        c_params[i] = [paramValues[i] doubleValue];
    }
    
    int result = chebyshev_approximator_init(&_approx, numCoeffs, numParams,
                                            c_params, timeScale);
    
    free(c_params);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"Chebyshev initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    chebyshev_approximator_free(&_approx);
}

- (NSUInteger)numCoefficients {
    return _approx.num_coefficients;
}

- (NSUInteger)numParams {
    return _approx.num_params;
}

- (BOOL)loadCoefficients:(NSArray<NSArray<NSNumber*>*>*)coefficients
                   error:(NSError**)error {
    if (!coefficients || coefficients.count != _approx.num_params) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid coefficients"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < _approx.num_params; i++) {
        NSArray<NSNumber*>* row = coefficients[i];
        if (row.count != _approx.num_coefficients) {
            if (error) {
                *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Invalid coefficient dimensions"}];
            }
            return NO;
        }
        
        for (NSUInteger j = 0; j < _approx.num_coefficients; j++) {
            _approx.coefficients[i][j] = [row[j] doubleValue];
        }
    }
    
    return YES;
}

- (BOOL)solveAtTime:(double)t
        paramIndex:(NSUInteger)paramIdx
          solution:(double*)solution
             error:(NSError**)error {
    if (!solution || paramIdx >= _approx.num_params) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    int result = chebyshev_approximator_solve(&_approx, t, paramIdx, solution);
    if (result != 0 && error) {
        *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"Chebyshev solve failed"}];
    }
    return result == 0;
}

@end

// ============================================================================
// Hybrid O(1) Solver
// ============================================================================

@implementation DDRKAMO1ApproximationSolver {
    O1ApproximationSolver _solver;
}

- (instancetype)initWithMethod:(DDRKAMO1Method)method
                         error:(NSError**)error {
    self = [super init];
    if (!self) return nil;
    
    int c_method = (int)method;
    int result = o1_solver_init(&_solver, c_method);
    
    if (result != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:result
                                     userInfo:@{NSLocalizedDescriptionKey: @"O(1) solver initialization failed"}];
        }
        return nil;
    }
    
    return self;
}

- (void)dealloc {
    o1_solver_free(&_solver);
}

- (DDRKAMO1Method)method {
    return (DDRKAMO1Method)_solver.method;
}

- (BOOL)useFallback {
    return _solver.use_fallback != 0;
}

- (void)setUseFallback:(BOOL)useFallback {
    _solver.use_fallback = useFallback ? 1 : 0;
}

- (BOOL)solveAtTime:(double)t
      initialCondition:(NSArray<NSNumber*>*)y0
            parameters:(NSArray<NSNumber*>*)params
            solution:(NSArray<NSNumber*>**)solution
               error:(NSError**)error {
    if (!y0 || !solution) {
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Invalid parameters"}];
        }
        return NO;
    }
    
    size_t state_dim = y0.count;
    double* c_y0 = (double*)malloc(state_dim * sizeof(double));
    double* c_params = NULL;
    double* c_solution = (double*)malloc(state_dim * sizeof(double));
    
    if (!c_y0 || !c_solution) {
        free(c_y0);
        free(c_solution);
        if (error) {
            *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Memory allocation failed"}];
        }
        return NO;
    }
    
    for (NSUInteger i = 0; i < state_dim; i++) {
        c_y0[i] = [y0[i] doubleValue];
    }
    
    size_t num_params = 0;
    if (params && params.count > 0) {
        num_params = params.count;
        c_params = (double*)malloc(num_params * sizeof(double));
        for (NSUInteger i = 0; i < num_params; i++) {
            c_params[i] = [params[i] doubleValue];
        }
    }
    
    int result = o1_solver_solve(&_solver, t, c_y0, c_params, num_params, c_solution);
    
    if (result == 0) {
        NSMutableArray<NSNumber*>* sol = [NSMutableArray arrayWithCapacity:state_dim];
        for (NSUInteger i = 0; i < state_dim; i++) {
            [sol addObject:@(c_solution[i])];
        }
        *solution = [sol copy];
    } else if (error) {
        *error = [NSError errorWithDomain:@"DDRKAMO1Approximation"
                                     code:result
                                 userInfo:@{NSLocalizedDescriptionKey: @"O(1) solve failed"}];
    }
    
    free(c_y0);
    free(c_params);
    free(c_solution);
    
    return result == 0;
}

@end
