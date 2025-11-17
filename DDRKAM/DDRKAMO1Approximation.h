/*
 * DDRKAM O(1) Approximation Solvers - Objective-C Interface
 * Real-Time O(1) Approximation Methods for Differential Equations
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * O(1) Approximation Method
 */
typedef NS_ENUM(NSInteger, DDRKAMO1Method) {
    DDRKAMO1MethodLookupTable = 0,
    DDRKAMO1MethodNeuralNetwork = 1,
    DDRKAMO1MethodChebyshev = 2,
    DDRKAMO1MethodReducedOrder = 3,
    DDRKAMO1MethodAuto = 4
};

/**
 * Lookup Table O(1) Solver
 * Pre-computed solutions with O(1) lookup
 */
@interface DDRKAMLookupTableSolver : NSObject

@property (nonatomic, readonly) NSUInteger paramGridSize;
@property (nonatomic, readonly) NSUInteger timeGridSize;
@property (nonatomic, readonly) NSUInteger stateDimension;

- (instancetype)initWithParamGridSize:(NSUInteger)paramGridSize
                         timeGridSize:(NSUInteger)timeGridSize
                        stateDimension:(NSUInteger)stateDim
                            paramMin:(double)paramMin
                            paramMax:(double)paramMax
                             timeMax:(double)timeMax
                               error:(NSError**)error;

- (BOOL)precomputeWithODEFunction:(void(*)(double t, const double* y, double* dydt, void* params))odeFunc
                            params:(void* _Nullable)params
                             error:(NSError**)error;

- (BOOL)solveAtTime:(double)t
            parameter:(double)param
      initialCondition:(NSArray<NSNumber*>*)y0
            solution:(NSArray<NSNumber*>**)solution
               error:(NSError**)error;

@end

/**
 * Neural Network O(1) Approximator
 */
@interface DDRKAMNeuralApproximator : NSObject

@property (nonatomic, readonly) NSUInteger inputDimension;
@property (nonatomic, readonly) NSUInteger outputDimension;
@property (nonatomic, readonly) NSUInteger numLayers;

- (instancetype)initWithLayerSizes:(NSArray<NSNumber*>*)layerSizes
                             error:(NSError**)error;

- (BOOL)loadWeights:(NSArray<NSArray<NSNumber*>*>*)weights
            biases:(NSArray<NSNumber*>*)biases
             error:(NSError**)error;

- (BOOL)solveAtTime:(double)t
      initialCondition:(NSArray<NSNumber*>*)y0
            parameters:(NSArray<NSNumber*>*)params
            solution:(NSArray<NSNumber*>**)solution
               error:(NSError**)error;

@end

/**
 * Chebyshev Polynomial O(1) Approximator
 */
@interface DDRKAMChebyshevApproximator : NSObject

@property (nonatomic, readonly) NSUInteger numCoefficients;
@property (nonatomic, readonly) NSUInteger numParams;

- (instancetype)initWithNumCoefficients:(NSUInteger)numCoeffs
                              numParams:(NSUInteger)numParams
                          paramValues:(NSArray<NSNumber*>*)paramValues
                            timeScale:(double)timeScale
                                error:(NSError**)error;

- (BOOL)loadCoefficients:(NSArray<NSArray<NSNumber*>*>*)coefficients
                   error:(NSError**)error;

- (BOOL)solveAtTime:(double)t
        paramIndex:(NSUInteger)paramIdx
          solution:(double*)solution
             error:(NSError**)error;

@end

/**
 * Hybrid O(1) Approximation Solver
 */
@interface DDRKAMO1ApproximationSolver : NSObject

@property (nonatomic, readonly) DDRKAMO1Method method;
@property (nonatomic) BOOL useFallback;

- (instancetype)initWithMethod:(DDRKAMO1Method)method
                         error:(NSError**)error;

- (BOOL)solveAtTime:(double)t
      initialCondition:(NSArray<NSNumber*>*)y0
            parameters:(NSArray<NSNumber*>*)params
            solution:(NSArray<NSNumber*>**)solution
               error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
