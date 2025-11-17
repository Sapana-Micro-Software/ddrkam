/*
 * DDRKAM Quantum ODE Solver - Objective-C Interface
 * Quantum-inspired nonlinear ODE solver for post-real-time future prediction
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Quantum parameters for quantum-inspired optimization
 */
@interface DDRKAMQuantumParams : NSObject

@property (nonatomic) double temperature;
@property (nonatomic) double tunnelingStrength;
@property (nonatomic) double coherenceTime;
@property (nonatomic) NSUInteger numIterations;
@property (nonatomic) double convergenceThreshold;

- (instancetype)initWithTemperature:(double)temperature
                   tunnelingStrength:(double)tunnelingStrength
                      coherenceTime:(double)coherenceTime
                       numIterations:(NSUInteger)numIterations
              convergenceThreshold:(double)convergenceThreshold;

@end

/**
 * Quantum ODE Solver
 * Uses quantum-inspired methods for nonlinear ODE solving and future prediction
 */
@interface DDRKAMQuantumODESolver : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) NSUInteger numQuantumStates;
@property (nonatomic, readonly) NSUInteger predictionHorizon;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) NSUInteger totalSteps;
@property (nonatomic, readonly) uint64_t quantumOperations;
@property (nonatomic, readonly) double predictionConfidence;
@property (nonatomic) BOOL usePostRealtime;

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
                                 error:(NSError**)error;

- (BOOL)stepAtTime:(double)t
              state:(NSArray<NSNumber*>*)y
              error:(NSError**)error;

- (BOOL)predictFutureAtTime:(double)t
                 currentState:(NSArray<NSNumber*>*)yCurrent
                futureStates:(NSArray<NSArray<NSNumber*>*>**)futureStates
                  confidence:(double*)confidence
                       error:(NSError**)error;

- (BOOL)refineAtTime:(double)t
         initialSolution:(NSArray<NSNumber*>*)yInitial
          refinedSolution:(NSArray<NSNumber*>**)yRefined
         maxIterations:(NSUInteger)maxIterations
                  error:(NSError**)error;

- (BOOL)solveFromTime:(double)t0
               toTime:(double)tEnd
        initialCondition:(NSArray<NSNumber*>*)y0
          numSteps:(NSUInteger)numSteps
      solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
             error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
