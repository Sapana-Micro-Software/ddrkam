/*
 * DDRKAM Causal and Granger Causality ODE Solvers - Objective-C Interface
 * Real-Time Causal RK4, Adams, and Granger Causality Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Causal RK4 Solver
 * Real-time RK4 that only uses past information (strictly causal)
 */
@interface DDRKAMCausalRK4Solver : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) NSUInteger totalSteps;
@property (nonatomic, readonly) BOOL strictCausality;

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                          historySize:(NSUInteger)historySize
                       strictCausality:(BOOL)strictCausality
                                 error:(NSError**)error;

- (BOOL)stepWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                      atTime:(double)t
                       state:(NSArray<NSNumber*>*)y
                      params:(void* _Nullable)params
                       error:(NSError**)error;

- (BOOL)solveWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                    fromTime:(double)t0
                      toTime:(double)tEnd
           initialCondition:(NSArray<NSNumber*>*)y0
                      params:(void* _Nullable)params
                 numSteps:(NSUInteger)numSteps
             solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                     error:(NSError**)error;

@end

/**
 * Causal Adams Method Solver
 * Real-time Adams that only uses past information
 */
@interface DDRKAMCausalAdamsSolver : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) NSUInteger adamsOrder;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) NSUInteger totalSteps;
@property (nonatomic, readonly) BOOL strictCausality;

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                           adamsOrder:(NSUInteger)adamsOrder
                          historySize:(NSUInteger)historySize
                       strictCausality:(BOOL)strictCausality
                                 error:(NSError**)error;

- (BOOL)stepWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                      atTime:(double)t
                       state:(NSArray<NSNumber*>*)y
                      params:(void* _Nullable)params
                       error:(NSError**)error;

- (BOOL)solveWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                    fromTime:(double)t0
                      toTime:(double)tEnd
           initialCondition:(NSArray<NSNumber*>*)y0
                      params:(void* _Nullable)params
                 numSteps:(NSUInteger)numSteps
             solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                     error:(NSError**)error;

@end

/**
 * Granger Causality Solver
 * Analyzes causal relationships and adapts solving strategy
 */
typedef NS_ENUM(NSInteger, DDRKAMGrangerBaseMethod) {
    DDRKAMGrangerBaseRK4 = 0,
    DDRKAMGrangerBaseAdams = 1
};

@interface DDRKAMGrangerCausalitySolver : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) DDRKAMGrangerBaseMethod baseMethod;
@property (nonatomic, readonly) NSUInteger causalityWindow;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) NSUInteger totalSteps;
@property (nonatomic, readonly) uint64_t causalityUpdates;

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                           baseMethod:(DDRKAMGrangerBaseMethod)baseMethod
                           adamsOrder:(NSUInteger)adamsOrder
                      causalityWindow:(NSUInteger)causalityWindow
                                 error:(NSError**)error;

- (BOOL)stepWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                      atTime:(double)t
                       state:(NSArray<NSNumber*>*)y
                      params:(void* _Nullable)params
                       error:(NSError**)error;

- (BOOL)updateCausalityWithError:(NSError**)error;

- (BOOL)getCausalityMatrix:(NSArray<NSArray<NSNumber*>*>**)causalityMatrix
                      error:(NSError**)error;

- (BOOL)getVariableImportance:(NSArray<NSNumber*>**)importance
                         error:(NSError**)error;

- (BOOL)solveWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                    fromTime:(double)t0
                      toTime:(double)tEnd
           initialCondition:(NSArray<NSNumber*>*)y0
                      params:(void* _Nullable)params
                 numSteps:(NSUInteger)numSteps
             solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                     error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
