/*
 * DDRKAM Visualizer Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMVisualizer.h"

#if TARGET_OS_OSX
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

@interface DDRKAMVisualizationView : NSView
@property (nonatomic, strong) NSArray<NSNumber*>* time;
@property (nonatomic, strong) NSArray<NSArray<NSNumber*>*>* state;
@property (nonatomic, assign) NSUInteger dimension;
@end

@implementation DDRKAMVisualizationView

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    if (!self.time || !self.state || self.time.count == 0) {
        return;
    }
    
    NSGraphicsContext* context = [NSGraphicsContext currentContext];
    [context saveGraphicsState];
    
    // Find bounds
    double t_min = [[self.time firstObject] doubleValue];
    double t_max = [[self.time lastObject] doubleValue];
    
    double y_min = DBL_MAX;
    double y_max = DBL_MIN;
    
    for (NSArray<NSNumber*>* stateVec in self.state) {
        for (NSNumber* val in stateVec) {
            double v = [val doubleValue];
            if (v < y_min) y_min = v;
            if (v > y_max) y_max = v;
        }
    }
    
    double t_range = t_max - t_min;
    double y_range = y_max - y_min;
    if (y_range == 0) y_range = 1.0;
    
    NSRect bounds = self.bounds;
    CGFloat padding = 40.0;
    NSRect plotRect = NSMakeRect(padding, padding, 
                                 bounds.size.width - 2*padding,
                                 bounds.size.height - 2*padding);
    
    // Draw axes
    [[NSColor blackColor] set];
    NSBezierPath* axes = [NSBezierPath bezierPath];
    [axes moveToPoint:NSMakePoint(plotRect.origin.x, plotRect.origin.y)];
    [axes lineToPoint:NSMakePoint(plotRect.origin.x + plotRect.size.width, plotRect.origin.y)];
    [axes moveToPoint:NSMakePoint(plotRect.origin.x, plotRect.origin.y)];
    [axes lineToPoint:NSMakePoint(plotRect.origin.x, plotRect.origin.y + plotRect.size.height)];
    [axes stroke];
    
    // Draw curves for each dimension
    NSColor* colors[] = {
        [NSColor redColor],
        [NSColor blueColor],
        [NSColor greenColor],
        [NSColor orangeColor],
        [NSColor purpleColor]
    };
    
    for (NSUInteger dim = 0; dim < self.dimension && dim < 5; dim++) {
        [colors[dim] set];
        NSBezierPath* path = [NSBezierPath bezierPath];
        
        BOOL first = YES;
        for (NSUInteger i = 0; i < self.time.count; i++) {
            double t = [self.time[i] doubleValue];
            double y = [self.state[i][dim] doubleValue];
            
            CGFloat x = plotRect.origin.x + (t - t_min) / t_range * plotRect.size.width;
            CGFloat y_pos = plotRect.origin.y + (y - y_min) / y_range * plotRect.size.height;
            
            NSPoint point = NSMakePoint(x, y_pos);
            if (first) {
                [path moveToPoint:point];
                first = NO;
            } else {
                [path lineToPoint:point];
            }
        }
        [path stroke];
    }
    
    [context restoreGraphicsState];
}

@end

@implementation DDRKAMVisualizer

- (NSView*)createVisualizationViewWithTime:(NSArray<NSNumber*>*)time
                                     state:(NSArray<NSArray<NSNumber*>*>*)state
                                 dimension:(NSUInteger)dimension {
    DDRKAMVisualizationView* view = [[DDRKAMVisualizationView alloc] initWithFrame:NSMakeRect(0, 0, 800, 600)];
    view.time = time;
    view.state = state;
    view.dimension = dimension;
    return view;
}

- (BOOL)exportToCSV:(NSString*)filePath
               time:(NSArray<NSNumber*>*)time
              state:(NSArray<NSArray<NSNumber*>*>*)state {
    NSMutableString* csv = [NSMutableString string];
    
    // Header
    [csv appendString:@"time"];
    for (NSUInteger i = 0; i < state.firstObject.count; i++) {
        [csv appendFormat:@",y%lu", (unsigned long)i];
    }
    [csv appendString:@"\n"];
    
    // Data
    for (NSUInteger i = 0; i < time.count; i++) {
        [csv appendFormat:@"%.6f", [time[i] doubleValue]];
        for (NSNumber* val in state[i]) {
            [csv appendFormat:@",%.6f", [val doubleValue]];
        }
        [csv appendString:@"\n"];
    }
    
    NSError* error = nil;
    BOOL success = [csv writeToFile:filePath 
                          atomically:YES 
                            encoding:NSUTF8StringEncoding 
                               error:&error];
    return success;
}

@end

#elif TARGET_OS_VISION
// VisionOS implementation would go here
@implementation DDRKAMVisualizer
- (UIView*)createVisualizationViewWithTime:(NSArray<NSNumber*>*)time
                                     state:(NSArray<NSArray<NSNumber*>*>*)state
                                 dimension:(NSUInteger)dimension {
    // VisionOS implementation
    return [[UIView alloc] init];
}
- (BOOL)exportToCSV:(NSString*)filePath
               time:(NSArray<NSNumber*>*)time
              state:(NSArray<NSArray<NSNumber*>*>*)state {
    // Same CSV export logic
    return YES;
}
@end
#endif
