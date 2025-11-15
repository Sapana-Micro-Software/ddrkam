/*
 * DDRKAM Solver Interface
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * ODE function block type
 * @param t Current time
 * @param y Current state vector
 * @param dydt Output derivative vector
 * @param params User-defined parameters
 */
typedef void (^ODEFunctionBlock)(double t, const double* y, double* dydt, void* _Nullable params);

/**
 * DDRKAM Solver using Runge-Kutta 3rd order method
 */
@interface DDRKAMSolver : NSObject

/**
 * Initialize solver with system dimension
 */
- (instancetype)initWithDimension:(NSUInteger)dimension;

/**
 * Solve ODE system
 * @param f ODE function block
 * @param t0 Initial time
 * @param tEnd Final time
 * @param y0 Initial state vector
 * @param stepSize Step size
 * @param params Optional parameters
 * @return Dictionary with "time" and "state" arrays
 */
- (NSDictionary<NSString*, id>*)solveWithFunction:(ODEFunctionBlock)f
                                         startTime:(double)t0
                                           endTime:(double)tEnd
                                       initialState:(NSArray<NSNumber*>*)y0
                                          stepSize:(double)stepSize
                                           params:(void* _Nullable)params;

/**
 * Single step integration
 */
- (NSArray<NSNumber*>*)stepWithFunction:(ODEFunctionBlock)f
                                  time:(double)t
                                 state:(NSArray<NSNumber*>*)y
                              stepSize:(double)h
                                params:(void* _Nullable)params;

@end

NS_ASSUME_NONNULL_END
