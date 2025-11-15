/*
 * DDRKAM Visualizer
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>
#if TARGET_OS_OSX
#import <AppKit/AppKit.h>
#elif TARGET_OS_VISION
#import <UIKit/UIKit.h>
#endif

NS_ASSUME_NONNULL_BEGIN

/**
 * Visualization component for ODE solutions
 */
@interface DDRKAMVisualizer : NSObject

/**
 * Create visualization view
 */
#if TARGET_OS_OSX
- (NSView*)createVisualizationViewWithTime:(NSArray<NSNumber*>*)time
                                     state:(NSArray<NSArray<NSNumber*>*>*)state
                                 dimension:(NSUInteger)dimension;
#elif TARGET_OS_VISION
- (UIView*)createVisualizationViewWithTime:(NSArray<NSNumber*>*)time
                                     state:(NSArray<NSArray<NSNumber*>*>*)state
                                 dimension:(NSUInteger)dimension;
#endif

/**
 * Export solution to CSV
 */
- (BOOL)exportToCSV:(NSString*)filePath
               time:(NSArray<NSNumber*>*)time
              state:(NSArray<NSArray<NSNumber*>*>*)state;

@end

NS_ASSUME_NONNULL_END
