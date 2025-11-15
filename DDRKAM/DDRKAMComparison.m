/*
 * DDRKAM Comparison Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMComparison.h"
#import "../include/comparison.h"

@implementation DDRKAMComparisonResults

- (NSString*)description {
    return [NSString stringWithFormat:
            @"RK3: time=%.6fs, error=%.6e, accuracy=%.2f%%, steps=%lu\n"
            @"DDRK3: time=%.6fs, error=%.6e, accuracy=%.2f%%, steps=%lu\n"
            @"AM: time=%.6fs, error=%.6e, accuracy=%.2f%%, steps=%lu\n"
            @"DDAM: time=%.6fs, error=%.6e, accuracy=%.2f%%, steps=%lu",
            self.rk3Time, self.rk3Error, self.rk3Accuracy * 100, (unsigned long)self.rk3Steps,
            self.ddrk3Time, self.ddrk3Error, self.ddrk3Accuracy * 100, (unsigned long)self.ddrk3Steps,
            self.amTime, self.amError, self.amAccuracy * 100, (unsigned long)self.amSteps,
            self.ddamTime, self.ddamError, self.ddamAccuracy * 100, (unsigned long)self.ddamSteps];
}

- (NSDictionary<NSString*, id>*)toDictionary {
    return @{
        @"RK3": @{
            @"time": @(self.rk3Time),
            @"error": @(self.rk3Error),
            @"accuracy": @(self.rk3Accuracy),
            @"steps": @(self.rk3Steps)
        },
        @"DDRK3": @{
            @"time": @(self.ddrk3Time),
            @"error": @(self.ddrk3Error),
            @"accuracy": @(self.ddrk3Accuracy),
            @"steps": @(self.ddrk3Steps)
        },
        @"AM": @{
            @"time": @(self.amTime),
            @"error": @(self.amError),
            @"accuracy": @(self.amAccuracy),
            @"steps": @(self.amSteps)
        },
        @"DDAM": @{
            @"time": @(self.ddamTime),
            @"error": @(self.ddamError),
            @"accuracy": @(self.ddamAccuracy),
            @"steps": @(self.ddamSteps)
        }
    };
}

@end

@implementation DDRKAMComparison

static void comparison_ode_wrapper(double t, const double* y, double* dydt, void* params) {
    NSDictionary* context = (__bridge NSDictionary*)params;
    ODEFunctionBlock block = context[@"function"];
    void* userParams = (__bridge void*)context[@"params"];
    
    if (block) {
        block(t, y, dydt, userParams);
    }
}

+ (DDRKAMComparisonResults*)compareMethodsWithFunction:(ODEFunctionBlock)f
                                              startTime:(double)t0
                                                endTime:(double)tEnd
                                            initialState:(NSArray<NSNumber*>*)y0
                                               stepSize:(double)stepSize
                                          exactSolution:(NSArray<NSNumber*>*)exactSolution
                                                 params:(void* _Nullable)params {
    if (!f || !y0 || !exactSolution || y0.count != exactSolution.count) {
        return nil;
    }
    
    size_t n = y0.count;
    double* y0_array = (double*)malloc(n * sizeof(double));
    double* exact_array = (double*)malloc(n * sizeof(double));
    
    if (!y0_array || !exact_array) {
        if (y0_array) free(y0_array);
        if (exact_array) free(exact_array);
        return nil;
    }
    
    for (NSUInteger i = 0; i < n; i++) {
        y0_array[i] = [y0[i] doubleValue];
        exact_array[i] = [exactSolution[i] doubleValue];
    }
    
    NSDictionary* context = @{
        @"function": f,
        @"params": params ? [NSValue valueWithPointer:params] : [NSNull null]
    };
    
    ComparisonResults c_results;
    if (compare_methods(comparison_ode_wrapper, t0, tEnd, y0_array, n, stepSize,
                       (__bridge void*)context, exact_array, &c_results) != 0) {
        free(y0_array);
        free(exact_array);
        return nil;
    }
    
    DDRKAMComparisonResults* results = [[DDRKAMComparisonResults alloc] init];
    [results setValue:@(c_results.rk3_time) forKey:@"rk3Time"];
    [results setValue:@(c_results.ddrk3_time) forKey:@"ddrk3Time"];
    [results setValue:@(c_results.am_time) forKey:@"amTime"];
    [results setValue:@(c_results.ddam_time) forKey:@"ddamTime"];
    
    [results setValue:@(c_results.rk3_error) forKey:@"rk3Error"];
    [results setValue:@(c_results.ddrk3_error) forKey:@"ddrk3Error"];
    [results setValue:@(c_results.am_error) forKey:@"amError"];
    [results setValue:@(c_results.ddam_error) forKey:@"ddamError"];
    
    [results setValue:@(c_results.rk3_accuracy) forKey:@"rk3Accuracy"];
    [results setValue:@(c_results.ddrk3_accuracy) forKey:@"ddrk3Accuracy"];
    [results setValue:@(c_results.am_accuracy) forKey:@"amAccuracy"];
    [results setValue:@(c_results.ddam_accuracy) forKey:@"ddamAccuracy"];
    
    [results setValue:@(c_results.rk3_steps) forKey:@"rk3Steps"];
    [results setValue:@(c_results.ddrk3_steps) forKey:@"ddrk3Steps"];
    [results setValue:@(c_results.am_steps) forKey:@"amSteps"];
    [results setValue:@(c_results.ddam_steps) forKey:@"ddamSteps"];
    
    free(y0_array);
    free(exact_array);
    
    return results;
}

+ (BOOL)exportResults:(DDRKAMComparisonResults*)results toCSV:(NSString*)filePath {
    if (!results || !filePath) {
        return NO;
    }
    
    NSMutableString* csv = [NSMutableString string];
    [csv appendString:@"Method,Time(s),Steps,Error,Accuracy(%)\n"];
    [csv appendFormat:@"RK3,%.6f,%lu,%.6e,%.6f\n",
     results.rk3Time, (unsigned long)results.rk3Steps, results.rk3Error, results.rk3Accuracy * 100];
    [csv appendFormat:@"DDRK3,%.6f,%lu,%.6e,%.6f\n",
     results.ddrk3Time, (unsigned long)results.ddrk3Steps, results.ddrk3Error, results.ddrk3Accuracy * 100];
    [csv appendFormat:@"AM,%.6f,%lu,%.6e,%.6f\n",
     results.amTime, (unsigned long)results.amSteps, results.amError, results.amAccuracy * 100];
    [csv appendFormat:@"DDAM,%.6f,%lu,%.6e,%.6f\n",
     results.ddamTime, (unsigned long)results.ddamSteps, results.ddamError, results.ddamAccuracy * 100];
    
    NSError* error = nil;
    BOOL success = [csv writeToFile:filePath atomically:YES encoding:NSUTF8StringEncoding error:&error];
    return success;
}

@end
