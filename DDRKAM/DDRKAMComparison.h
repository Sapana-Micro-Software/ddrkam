/*
 * DDRKAM Comparison Interface (Objective-C)
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Comparison results structure
 */
@interface DDRKAMComparisonResults : NSObject

@property (nonatomic, readonly) double rk3Time;
@property (nonatomic, readonly) double ddrk3Time;
@property (nonatomic, readonly) double amTime;
@property (nonatomic, readonly) double ddamTime;

@property (nonatomic, readonly) double rk3Error;
@property (nonatomic, readonly) double ddrk3Error;
@property (nonatomic, readonly) double amError;
@property (nonatomic, readonly) double ddamError;

@property (nonatomic, readonly) double rk3Accuracy;
@property (nonatomic, readonly) double ddrk3Accuracy;
@property (nonatomic, readonly) double amAccuracy;
@property (nonatomic, readonly) double ddamAccuracy;

@property (nonatomic, readonly) NSUInteger rk3Steps;
@property (nonatomic, readonly) NSUInteger ddrk3Steps;
@property (nonatomic, readonly) NSUInteger amSteps;
@property (nonatomic, readonly) NSUInteger ddamSteps;

- (NSString*)description;
- (NSDictionary<NSString*, id>*)toDictionary;

@end

/**
 * Comparison runner
 */
@interface DDRKAMComparison : NSObject

/**
 * Compare all methods (RK3, DDRK3, AM, DDAM)
 * 
 * @param f ODE function block
 * @param t0 Initial time
 * @param tEnd Final time
 * @param y0 Initial state
 * @param stepSize Step size
 * @param exactSolution Exact solution at tEnd
 * @param params Optional parameters
 * @return Comparison results
 */
+ (nullable DDRKAMComparisonResults*)compareMethodsWithFunction:(ODEFunctionBlock)f
                                                        startTime:(double)t0
                                                          endTime:(double)tEnd
                                                      initialState:(NSArray<NSNumber*>*)y0
                                                         stepSize:(double)stepSize
                                                    exactSolution:(NSArray<NSNumber*>*)exactSolution
                                                           params:(void* _Nullable)params;

/**
 * Export results to CSV
 */
+ (BOOL)exportResults:(DDRKAMComparisonResults*)results toCSV:(NSString*)filePath;

@end

NS_ASSUME_NONNULL_END
