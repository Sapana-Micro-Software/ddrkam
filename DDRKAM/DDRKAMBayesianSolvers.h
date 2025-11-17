/*
 * DDRKAM Bayesian ODE Solvers - Objective-C Interface
 * Real-Time Bayesian and Randomized DP Solvers
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Bayesian solver mode
 */
typedef NS_ENUM(NSInteger, DDRKAMBayesianMode) {
    DDRKAMBayesianModeProbabilistic = 0,
    DDRKAMBayesianModeExact = 1,
    DDRKAMBayesianModeHybrid = 2
};

/**
 * Forward-Backward probabilistic solver
 */
@interface DDRKAMForwardBackwardSolver : NSObject

@property (nonatomic, readonly) NSUInteger stateSpaceSize;
@property (nonatomic, readonly) double observationNoiseVariance;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) NSUInteger currentStep;

- (instancetype)initWithStateSpaceSize:(NSUInteger)stateSpaceSize
                         stateValues:(NSArray<NSNumber*>*)stateValues
                    transitionMatrix:(NSArray<NSArray<NSNumber*>*>*)transitionMatrix
                               prior:(NSArray<NSNumber*>*)prior
            observationNoiseVariance:(double)noiseVariance;

- (BOOL)stepWithObservation:(double)observation error:(NSError**)error;
- (BOOL)computePosteriorWithError:(NSError**)error;
- (BOOL)getStatisticsMean:(double*)mean
                  variance:(double*)variance
            fullPosterior:(NSArray<NSNumber*>** _Nullable)fullPosterior
                    error:(NSError**)error;

@end

/**
 * Viterbi exact (MAP) solver
 */
@interface DDRKAMViterbiSolver : NSObject

@property (nonatomic, readonly) NSUInteger stateSpaceSize;
@property (nonatomic, readonly) double currentTime;
@property (nonatomic, readonly) double mapProbability;

- (instancetype)initWithStateSpaceSize:(NSUInteger)stateSpaceSize
                          stateValues:(NSArray<NSNumber*>*)stateValues
                     transitionMatrix:(NSArray<NSArray<NSNumber*>*>*)transitionMatrix
                                prior:(NSArray<NSNumber*>*)prior
             observationNoiseVariance:(double)noiseVariance;

- (BOOL)stepWithObservation:(double)observation error:(NSError**)error;
- (BOOL)getMAPEstimate:(double*)mapEstimate
         mapProbability:(double* _Nullable)mapProbability
                 error:(NSError**)error;

@end

/**
 * Randomized Dynamic Programming solver
 */
@interface DDRKAMRandomizedDPSolver : NSObject

@property (nonatomic, readonly) NSUInteger stateDimension;
@property (nonatomic, readonly) NSUInteger numSamples;
@property (nonatomic, readonly) NSUInteger numControls;
@property (nonatomic, readonly) double samplingRadius;
@property (nonatomic, readonly) double ucbConstant;
@property (nonatomic, readonly) uint64_t stepCount;

- (instancetype)initWithStateDimension:(NSUInteger)stateDim
                           numSamples:(NSUInteger)numSamples
                         numControls:(NSUInteger)numControls
                   controlCandidates:(NSArray<NSNumber*>*)controlCandidates
                             odeFunc:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                           odeParams:(void* _Nullable)odeParams
                       costFunction:(double(*)(double t, const double* y, double u, void* params))costFunc
                        costParams:(void* _Nullable)costParams
                    samplingRadius:(double)radius
                       ucbConstant:(double)ucbConstant
                             error:(NSError**)error;

- (BOOL)stepAtTime:(double)t
      currentState:(NSArray<NSNumber*>*)yCurrent
           nextState:(NSArray<NSNumber*>**)yNext
    optimalControl:(double*)optimalControl
             error:(NSError**)error;

- (BOOL)solveFromTime:(double)t0
               toTime:(double)tEnd
        initialCondition:(NSArray<NSNumber*>*)y0
          numSteps:(NSUInteger)numSteps
      solutionPath:(NSArray<NSArray<NSNumber*>*>**)solutionPath
          controls:(NSArray<NSNumber*>** _Nullable)controls
             error:(NSError**)error;

- (BOOL)getValueAtTime:(double)t
                 state:(NSArray<NSNumber*>*)y
      valueEstimate:(double*)valueEstimate
      valueVariance:(double* _Nullable)valueVariance
               error:(NSError**)error;

@end

/**
 * Real-Time Bayesian solver (hybrid)
 */
@interface DDRKAMRealTimeBayesianSolver : NSObject

@property (nonatomic, readonly) DDRKAMBayesianMode mode;
@property (nonatomic, readonly) NSUInteger stateSpaceSize;
@property (nonatomic, readonly) double currentTime;

- (instancetype)initWithMode:(DDRKAMBayesianMode)mode
              stateSpaceSize:(NSUInteger)stateSpaceSize
                 stateValues:(NSArray<NSNumber*>*)stateValues
            transitionMatrix:(NSArray<NSArray<NSNumber*>*>*)transitionMatrix
                       prior:(NSArray<NSNumber*>*)prior
    observationNoiseVariance:(double)noiseVariance
                       error:(NSError**)error;

- (BOOL)stepAtTime:(double)t
      observation:(double)observation
            yOut:(double*)yOut
            error:(NSError**)error;

- (BOOL)getStatisticsMean:(double*)mean
                 variance:(double*)variance
                     yMAP:(double* _Nullable)yMAP
                    error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
