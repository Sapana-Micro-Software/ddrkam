/*
 * DDRKAM Solver Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMSolver.h"
#import "../include/rk3.h"

@interface DDRKAMSolver ()
@property (nonatomic, assign) NSUInteger dimension;
@end

@implementation DDRKAMSolver

- (instancetype)initWithDimension:(NSUInteger)dimension {
    self = [super init];
    if (self) {
        _dimension = dimension;
    }
    return self;
}

static void ode_wrapper(double t, const double* y, double* dydt, void* params) {
    NSDictionary* context = (__bridge NSDictionary*)params;
    ODEFunctionBlock block = context[@"function"];
    void* userParams = (__bridge void*)context[@"params"];
    
    if (block) {
        block(t, y, dydt, userParams);
    }
}

- (NSArray<NSNumber*>*)stepWithFunction:(ODEFunctionBlock)f
                                  time:(double)t
                                 state:(NSArray<NSNumber*>*)y
                              stepSize:(double)h
                                params:(void* _Nullable)params {
    if (y.count != self.dimension) {
        return nil;
    }
    
    double* y_array = (double*)malloc(self.dimension * sizeof(double));
    if (!y_array) {
        return nil;
    }
    
    for (NSUInteger i = 0; i < self.dimension; i++) {
        y_array[i] = [y[i] doubleValue];
    }
    
    NSDictionary* context = @{
        @"function": f,
        @"params": params ? [NSValue valueWithPointer:params] : [NSNull null]
    };
    
    double t_new = rk3_step(ode_wrapper, t, y_array, self.dimension, h, 
                           (__bridge void*)context);
    
    NSMutableArray<NSNumber*>* result = [NSMutableArray arrayWithCapacity:self.dimension];
    for (NSUInteger i = 0; i < self.dimension; i++) {
        [result addObject:@(y_array[i])];
    }
    
    free(y_array);
    return result;
}

- (NSDictionary<NSString*, id>*)solveWithFunction:(ODEFunctionBlock)f
                                         startTime:(double)t0
                                           endTime:(double)tEnd
                                       initialState:(NSArray<NSNumber*>*)y0
                                          stepSize:(double)stepSize
                                           params:(void* _Nullable)params {
    if (y0.count != self.dimension) {
        return nil;
    }
    
    double* y0_array = (double*)malloc(self.dimension * sizeof(double));
    if (!y0_array) {
        return nil;
    }
    
    for (NSUInteger i = 0; i < self.dimension; i++) {
        y0_array[i] = [y0[i] doubleValue];
    }
    
    size_t max_steps = (size_t)((tEnd - t0) / stepSize) + 1;
    double* t_out = (double*)malloc(max_steps * sizeof(double));
    double* y_out = (double*)malloc(max_steps * self.dimension * sizeof(double));
    
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
    
    size_t num_steps = rk3_solve(ode_wrapper, t0, tEnd, y0_array, self.dimension, 
                                 stepSize, (__bridge void*)context, t_out, y_out);
    
    NSMutableArray<NSNumber*>* timeArray = [NSMutableArray arrayWithCapacity:num_steps];
    NSMutableArray<NSArray<NSNumber*>*>* stateArray = [NSMutableArray arrayWithCapacity:num_steps];
    
    for (size_t i = 0; i < num_steps; i++) {
        [timeArray addObject:@(t_out[i])];
        
        NSMutableArray<NSNumber*>* state = [NSMutableArray arrayWithCapacity:self.dimension];
        for (NSUInteger j = 0; j < self.dimension; j++) {
            [state addObject:@(y_out[i * self.dimension + j])];
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
