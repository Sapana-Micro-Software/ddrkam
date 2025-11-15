/*
 * DDRKAM PDE Solver Interface (Objective-C)
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * PDE Type
 */
typedef NS_ENUM(NSInteger, DDRKAMPDEType) {
    DDRKAMPDETypeHeat = 0,
    DDRKAMPDETypeWave,
    DDRKAMPDETypeLaplace,
    DDRKAMPDETypePoisson,
    DDRKAMPDETypeBurgers,
    DDRKAMPDETypeAdvection
};

/**
 * Spatial Dimension
 */
typedef NS_ENUM(NSInteger, DDRKAMSpatialDimension) {
    DDRKAMSpatialDimension1D = 1,
    DDRKAMSpatialDimension2D = 2,
    DDRKAMSpatialDimension3D = 3
};

/**
 * PDE Solution
 */
@interface DDRKAMPDESolution : NSObject

@property (nonatomic, readonly) NSArray<NSNumber*>* solution;
@property (nonatomic, readonly) NSArray<NSNumber*>* timeSteps;
@property (nonatomic, readonly) NSUInteger gridPoints;
@property (nonatomic, readonly) NSUInteger timeStepsCount;
@property (nonatomic, readonly) double currentTime;

@end

/**
 * PDE Solver
 */
@interface DDRKAMPDESolver : NSObject

/**
 * Solve 1D Heat Equation
 */
+ (nullable DDRKAMPDESolution*)solveHeat1DWithGridPoints:(NSUInteger)nx
                                               spatialStep:(double)dx
                                                 timeStep:(double)dt
                                            diffusionCoeff:(double)alpha
                                              initialCondition:(NSArray<NSNumber*>*)u0
                                                 finalTime:(double)tEnd;

/**
 * Solve 2D Heat Equation
 */
+ (nullable DDRKAMPDESolution*)solveHeat2DWithGridPointsX:(NSUInteger)nx
                                                      gridY:(NSUInteger)ny
                                                 spatialStepX:(double)dx
                                                 spatialStepY:(double)dy
                                                   timeStep:(double)dt
                                              diffusionCoeff:(double)alpha
                                            initialCondition:(NSArray<NSArray<NSNumber*>*>*)u0
                                                   finalTime:(double)tEnd;

/**
 * Solve 1D Wave Equation
 */
+ (nullable DDRKAMPDESolution*)solveWave1DWithGridPoints:(NSUInteger)nx
                                                spatialStep:(double)dx
                                                  timeStep:(double)dt
                                                 waveSpeed:(double)c
                                          initialCondition:(NSArray<NSNumber*>*)u0
                                                 finalTime:(double)tEnd;

/**
 * Solve 1D Advection Equation
 */
+ (nullable DDRKAMPDESolution*)solveAdvection1DWithGridPoints:(NSUInteger)nx
                                                    spatialStep:(double)dx
                                                      timeStep:(double)dt
                                                  advectionSpeed:(double)a
                                               initialCondition:(NSArray<NSNumber*>*)u0
                                                      finalTime:(double)tEnd;

/**
 * Export solution to CSV
 */
+ (BOOL)exportSolution:(DDRKAMPDESolution*)solution
            toFile:(NSString*)filePath
         dimension:(DDRKAMSpatialDimension)dim
         gridSizeX:(NSUInteger)nx
         gridSizeY:(NSUInteger)ny;

@end

NS_ASSUME_NONNULL_END
