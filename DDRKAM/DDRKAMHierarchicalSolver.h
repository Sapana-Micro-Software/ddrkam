/*
 * DDRKAM Hierarchical Solver
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>
#import "DDRKAMSolver.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Hierarchical/Transformer-like ODE Solver
 */
@interface DDRKAMHierarchicalSolver : NSObject

/**
 * Initialize with configuration
 */
- (instancetype)initWithDimension:(NSUInteger)dimension
                        numLayers:(NSUInteger)numLayers
                       hiddenDim:(NSUInteger)hiddenDim;

/**
 * Solve using hierarchical method
 */
- (NSDictionary<NSString*, id>*)solveWithFunction:(ODEFunctionBlock)f
                                         startTime:(double)t0
                                           endTime:(double)tEnd
                                       initialState:(NSArray<NSNumber*>*)y0
                                          stepSize:(double)stepSize
                                           params:(void* _Nullable)params;

@end

NS_ASSUME_NONNULL_END
