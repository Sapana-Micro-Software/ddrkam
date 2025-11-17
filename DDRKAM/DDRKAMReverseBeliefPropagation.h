/*
 * DDRKAM Reverse Belief Propagation - Objective-C Interface
 * Lossless Tracing for Backwards Uncertainty Propagation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Belief structure: represents uncertainty/confidence in state
 */
@interface DDRKAMBelief : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) double timestamp;
@property (nonatomic, readonly) NSArray<NSNumber*>* mean;
@property (nonatomic, readonly) NSArray<NSArray<NSNumber*>*>* covariance;
@property (nonatomic, readonly) NSArray<NSNumber*>* confidence;

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                                  mean:(NSArray<NSNumber*>*)mean
                            covariance:(NSArray<NSArray<NSNumber*>*>*)covariance
                                 error:(NSError**)error;

- (instancetype)initWithBelief:(DDRKAMBelief*)belief;

@end

/**
 * Reverse Belief Propagation Solver
 * Propagates beliefs backwards in time with lossless tracing
 */
@interface DDRKAMReverseBeliefPropagationSolver : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) double stepSize;
@property (nonatomic, readonly) NSUInteger traceCapacity;
@property (nonatomic, readonly) NSUInteger traceCount;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) NSUInteger forwardSteps;
@property (nonatomic, readonly) NSUInteger reverseSteps;
@property (nonatomic, readonly) uint64_t traceOperations;
@property (nonatomic) BOOL storeJacobian;
@property (nonatomic) BOOL storeSensitivity;

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                             stepSize:(double)stepSize
                        traceCapacity:(NSUInteger)traceCapacity
                            odeFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                              odeParams:(void* _Nullable)odeParams
                         jacobianFunction:(void(* _Nullable)(double t, const double* y, double** jacobian, void* params))jacobianFunc
                        jacobianParams:(void* _Nullable)jacobianParams
                         storeJacobian:(BOOL)storeJacobian
                        storeSensitivity:(BOOL)storeSensitivity
                                 error:(NSError**)error;

- (BOOL)forwardStepAtTime:(double)t
                     state:(NSArray<NSNumber*>*)y
                    belief:(DDRKAMBelief*)belief
                     error:(NSError**)error;

- (BOOL)reverseStepAtTime:(double)t
                     state:(NSArray<NSNumber*>**)y
                    belief:(DDRKAMBelief**)belief
                     error:(NSError**)error;

- (BOOL)forwardSolveFromTime:(double)t0
                       toTime:(double)tEnd
            initialCondition:(NSArray<NSNumber*>*)y0
                initialBelief:(DDRKAMBelief*)initialBelief
                       error:(NSError**)error;

- (BOOL)reverseSolveFromTime:(double)tStart
                       toTime:(double)tEnd
                  finalBelief:(DDRKAMBelief*)finalBelief
                     numSteps:(NSUInteger)numSteps
                 solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
                       beliefs:(NSArray<DDRKAMBelief*>**)beliefs
                        error:(NSError**)error;

- (BOOL)smoothAtTime:(double)t
         forwardState:(NSArray<NSNumber*>*)yForward
        forwardBelief:(DDRKAMBelief*)beliefForward
         reverseState:(NSArray<NSNumber*>*)yReverse
        reverseBelief:(DDRKAMBelief*)beliefReverse
         smoothedState:(NSArray<NSNumber*>**)ySmoothed
        smoothedBelief:(DDRKAMBelief**)beliefSmoothed
                error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
