/*
 * DDRKAM Hierarchical Solver Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMHierarchicalSolver.h"
#import "../include/hierarchical_rk.h"

@interface DDRKAMHierarchicalSolver ()
@property (nonatomic, assign) HierarchicalRKSolver* solver;
@end

@implementation DDRKAMHierarchicalSolver

- (instancetype)initWithDimension:(NSUInteger)dimension
                        numLayers:(NSUInteger)numLayers
                       hiddenDim:(NSUInteger)hiddenDim {
    self = [super init];
    if (self) {
        _solver = (HierarchicalRKSolver*)malloc(sizeof(HierarchicalRKSolver));
        if (_solver) {
            if (hierarchical_rk_init(_solver, numLayers, dimension, hiddenDim) != 0) {
                free(_solver);
                _solver = NULL;
                return nil;
            }
        }
    }
    return self;
}

- (void)dealloc {
    if (_solver) {
        hierarchical_rk_free(_solver);
        free(_solver);
    }
}

static void hierarchical_ode_wrapper(double t, const double* y, double* dydt, void* params) {
    NSDictionary* context = (__bridge NSDictionary*)params;
    ODEFunctionBlock block = context[@"function"];
    void* userParams = (__bridge void*)context[@"params"];
    
    if (block) {
        block(t, y, dydt, userParams);
    }
}

- (NSDictionary<NSString*, id>*)solveWithFunction:(ODEFunctionBlock)f
                                         startTime:(double)t0
                                           endTime:(double)tEnd
                                       initialState:(NSArray<NSNumber*>*)y0
                                          stepSize:(double)stepSize
                                           params:(void* _Nullable)params {
    if (!_solver || y0.count != _solver->state_dim) {
        return nil;
    }
    
    double* y0_array = (double*)malloc(_solver->state_dim * sizeof(double));
    if (!y0_array) {
        return nil;
    }
    
    for (NSUInteger i = 0; i < _solver->state_dim; i++) {
        y0_array[i] = [y0[i] doubleValue];
    }
    
    size_t max_steps = (size_t)((tEnd - t0) / stepSize) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * _solver->state_dim * sizeof(double));
    
    if (!t_out || !y_out) {
        free(y0_array);
        if (t_out) free(t_out);
        if (y_out) free(y_out);
        return nil;
    }
    
    NSDictionary* context = @{
        @"function": f,
        @"params": params ? [NSValue valueWithPointer:params] : [NSNull null]
    };
    
    size_t num_steps = hierarchical_rk_solve(_solver, hierarchical_ode_wrapper, 
                                            t0, tEnd, y0_array, stepSize,
                                            (__bridge void*)context, t_out, y_out);
    
    NSMutableArray<NSNumber*>* timeArray = [NSMutableArray arrayWithCapacity:num_steps];
    NSMutableArray<NSArray<NSNumber*>*>* stateArray = [NSMutableArray arrayWithCapacity:num_steps];
    
    for (size_t i = 0; i < num_steps; i++) {
        [timeArray addObject:@(t_out[i])];
        
        NSMutableArray<NSNumber*>* state = [NSMutableArray arrayWithCapacity:_solver->state_dim];
        for (NSUInteger j = 0; j < _solver->state_dim; j++) {
            [state addObject:@(y_out[i * _solver->state_dim + j])];
        }
        [stateArray addObject:state];
    }
    
    free(y0_array);
    free(t_out);
    free(y_out);
    
    return @{
        @"time": timeArray,
        @"state": stateArray
    };
}

@end
