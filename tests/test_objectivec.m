/*
 * Objective-C Benchmark Tests
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>
#import <DDRKAM/DDRKAMSolver.h>
#import <DDRKAM/DDRKAMHierarchicalSolver.h>
#import <DDRKAM/DDRKAMComparison.h>

void runObjectiveCBenchmarks() {
    @autoreleasepool {
        NSLog(@"=== Objective-C Benchmark Suite ===\n");
        
        // Test 1: Exponential Decay
        NSLog(@"Test 1: Exponential Decay");
        DDRKAMSolver* rk3Solver = [[DDRKAMSolver alloc] initWithDimension:1];
        
        NSArray<NSNumber*>* y0 = @[@1.0];
        NSArray<NSNumber*>* exact = @[@(exp(-2.0))];
        
        NSDate* start = [NSDate date];
        NSDictionary* result = [rk3Solver solveWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t;
            (void)params;
            dydt[0] = -y[0];
        } startTime:0.0 endTime:2.0 initialState:y0 stepSize:0.01 params:NULL];
        NSTimeInterval rk3Time = [[NSDate date] timeIntervalSinceDate:start];
        
        if (result) {
            NSArray<NSArray<NSNumber*>*>* states = result[@"state"];
            NSNumber* finalValue = states.lastObject[0];
            double error = fabs([finalValue doubleValue] - [exact[0] doubleValue]);
            double accuracy = 1.0 - error / fabs([exact[0] doubleValue]);
            
            NSLog(@"RK3: time=%.6fs, error=%.6e, accuracy=%.6f%%", 
                  rk3Time, error, accuracy * 100);
        }
        
        // Test DDRK3
        DDRKAMHierarchicalSolver* ddrk3Solver = [[DDRKAMHierarchicalSolver alloc] 
                                                 initWithDimension:1 numLayers:3 hiddenDim:16];
        
        start = [NSDate date];
        NSDictionary* ddrk3Result = [ddrk3Solver solveWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t;
            (void)params;
            dydt[0] = -y[0];
        } startTime:0.0 endTime:2.0 initialState:y0 stepSize:0.01 params:NULL];
        NSTimeInterval ddrk3Time = [[NSDate date] timeIntervalSinceDate:start];
        
        if (ddrk3Result) {
            NSArray<NSArray<NSNumber*>*>* states = ddrk3Result[@"state"];
            NSNumber* finalValue = states.lastObject[0];
            double error = fabs([finalValue doubleValue] - [exact[0] doubleValue]);
            double accuracy = 1.0 - error / fabs([exact[0] doubleValue]);
            
            NSLog(@"DDRK3: time=%.6fs, error=%.6e, accuracy=%.6f%%", 
                  ddrk3Time, error, accuracy * 100);
        }
        
        // Test Comparison
        NSLog(@"\nTest 2: Method Comparison");
        DDRKAMComparisonResults* compResults = [DDRKAMComparison compareMethodsWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t;
            (void)params;
            dydt[0] = -y[0];
        } startTime:0.0 endTime:2.0 initialState:y0 stepSize:0.01 exactSolution:exact params:NULL];
        
        if (compResults) {
            NSLog(@"%@", compResults);
            [DDRKAMComparison exportResults:compResults toCSV:@"objectivec_comparison.csv"];
            NSLog(@"✅ Comparison results exported to objectivec_comparison.csv");
        }
        
        NSLog(@"\n=== Objective-C Benchmarks Complete ===");
    }
}

int main(int argc, const char * argv[]) {
    runObjectiveCBenchmarks();
    return 0;
}
