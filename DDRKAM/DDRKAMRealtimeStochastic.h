/*
 * DDRKAM Real-Time and Stochastic Solvers (Objective-C)
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Stochastic Parameters
 */
@interface DDRKAMStochasticParams : NSObject

@property (nonatomic) double noiseAmplitude;
@property (nonatomic) double noiseCorrelation;
@property (nonatomic) BOOL useBrownianMotion;
@property (nonatomic) double randomSeed;

+ (instancetype)defaultParams;
+ (instancetype)paramsWithAmplitude:(double)amplitude useBrownian:(BOOL)useBrownian;

@end

/**
 * Real-Time Solver
 */
@interface DDRKAMRealtimeSolver : NSObject

@property (nonatomic, readonly) NSUInteger dimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) NSUInteger totalSteps;
@property (nonatomic, readonly) double currentTime;

/**
 * Initialize real-time RK3 solver
 */
- (instancetype)initWithDimension:(NSUInteger)n
                         stepSize:(double)h
                       bufferSize:(NSUInteger)bufferSize;

/**
 * Step real-time RK3 solver with streaming data
 */
- (BOOL)stepRK3WithFunction:(void (^)(double t, const double* y, double* dydt, void* params))f
                    newState:(NSArray<NSNumber*>*)yNew
                      params:(nullable void*)params
                    callback:(nullable void (^)(double t, NSArray<NSNumber*>* y))callback;

/**
 * Step real-time Adams solver
 */
- (BOOL)stepAdamsWithFunction:(void (^)(double t, const double* y, double* dydt, void* params))f
                      newState:(NSArray<NSNumber*>*)yNew
                        params:(nullable void*)params
                      callback:(nullable void (^)(double t, NSArray<NSNumber*>* y))callback;

@end

/**
 * Stochastic Solver
 */
@interface DDRKAMStochasticSolver : NSObject

@property (nonatomic, readonly) NSUInteger dimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) DDRKAMStochasticParams* params;

/**
 * Initialize stochastic RK3 solver
 */
- (instancetype)initRK3WithDimension:(NSUInteger)n
                             stepSize:(double)h
                                params:(DDRKAMStochasticParams*)params;

/**
 * Initialize stochastic Adams solver
 */
- (instancetype)initAdamsWithDimension:(NSUInteger)n
                              stepSize:(double)h
                                params:(DDRKAMStochasticParams*)params;

/**
 * Step stochastic RK3 solver
 */
- (double)stepRK3WithFunction:(void (^)(double t, const double* y, double* dydt, void* params))f
                      time:(double)t0
                      state:(NSMutableArray<NSNumber*>*)y0
                     params:(nullable void*)params;

/**
 * Step stochastic Adams solver
 */
- (double)stepAdamsWithFunction:(void (^)(double t, const double* y, double* dydt, void* params))f
                        time:(double)t0
                        state:(NSMutableArray<NSNumber*>*)y0
                       params:(nullable void*)params;

@end

/**
 * Data-Driven Adaptive Control
 */
@interface DDRKAMDataDrivenControl : NSObject

/**
 * Compute adaptive step size based on error history
 */
+ (double)adaptiveStepSizeWithErrorHistory:(NSArray<NSNumber*>*)errorHistory
                                 currentH:(double)currentH
                              targetError:(double)targetError;

/**
 * Select best method based on system characteristics
 */
+ (NSInteger)selectMethodWithStiffness:(double)stiffness
                          errorTolerance:(double)tolerance
                         speedRequirement:(double)speed;

@end

NS_ASSUME_NONNULL_END
