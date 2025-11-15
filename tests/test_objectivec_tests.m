/*
 * Objective-C Test Suite: Exponential Decay and Harmonic Oscillator
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import <Foundation/Foundation.h>
#import <DDRKAM/DDRKAMSolver.h>
#import <DDRKAM/DDRKAMHierarchicalSolver.h>
#import <DDRKAM/DDRKAMComparison.h>
#import <DDRKAM/DDRKAMAdams.h>

void testExponentialDecayObjectiveC() {
    @autoreleasepool {
        NSLog(@"╔══════════════════════════════════════════════════════════════╗");
        NSLog(@"║  Objective-C: Exponential Decay Test - All Methods           ║");
        NSLog(@"╚══════════════════════════════════════════════════════════════╝\n");
        
        NSLog(@"Test Case: dy/dt = -y, y(0) = 1.0, t ∈ [0, 2.0]");
        NSLog(@"Exact Solution: y(t) = exp(-t)");
        NSLog(@"Expected Final Value: y(2.0) = %.10f\n", exp(-2.0));
        
        NSArray<NSNumber*>* y0 = @[@1.0];
        NSArray<NSNumber*>* exact = @[@(exp(-2.0))];
        double t0 = 0.0;
        double tEnd = 2.0;
        double h = 0.01;
        
        // Test RK3
        NSLog(@"=== RK3 Method ===");
        DDRKAMSolver* rk3Solver = [[DDRKAMSolver alloc] initWithDimension:1];
        NSDate* start = [NSDate date];
        NSDictionary* rk3Result = [rk3Solver solveWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t; (void)params;
            dydt[0] = -y[0];
        } startTime:t0 endTime:tEnd initialState:y0 stepSize:h params:NULL];
        NSTimeInterval rk3Time = [[NSDate date] timeIntervalSinceDate:start];
        
        if (rk3Result) {
            NSArray<NSArray<NSNumber*>*>* states = rk3Result[@"state"];
            NSNumber* finalValue = states.lastObject[0];
            double error = fabs([finalValue doubleValue] - [exact[0] doubleValue]);
            double accuracy = (1.0 - error / fabs([exact[0] doubleValue])) * 100.0;
            NSLog(@"  Time: %.6f seconds", rk3Time);
            NSLog(@"  Final value: %.10f", [finalValue doubleValue]);
            NSLog(@"  Exact value: %.10f", [exact[0] doubleValue]);
            NSLog(@"  Error: %.6e", error);
            NSLog(@"  Accuracy: %.6f%%", accuracy);
            NSLog(@"  %@\n", (error < 1e-5) ? @"PASS" : @"FAIL");
        }
        
        // Test DDRK3
        NSLog(@"=== DDRK3 Method ===");
        DDRKAMHierarchicalSolver* ddrk3Solver = [[DDRKAMHierarchicalSolver alloc] 
                                                 initWithDimension:1 numLayers:3 hiddenDim:16];
        start = [NSDate date];
        NSDictionary* ddrk3Result = [ddrk3Solver solveWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t; (void)params;
            dydt[0] = -y[0];
        } startTime:t0 endTime:tEnd initialState:y0 stepSize:h params:NULL];
        NSTimeInterval ddrk3Time = [[NSDate date] timeIntervalSinceDate:start];
        
        if (ddrk3Result) {
            NSArray<NSArray<NSNumber*>*>* states = ddrk3Result[@"state"];
            NSNumber* finalValue = states.lastObject[0];
            double error = fabs([finalValue doubleValue] - [exact[0] doubleValue]);
            double accuracy = (1.0 - error / fabs([exact[0] doubleValue])) * 100.0;
            NSLog(@"  Time: %.6f seconds", ddrk3Time);
            NSLog(@"  Final value: %.10f", [finalValue doubleValue]);
            NSLog(@"  Error: %.6e", error);
            NSLog(@"  Accuracy: %.6f%%", accuracy);
            NSLog(@"  %@\n", (error < 1e-5) ? @"PASS" : @"FAIL");
        }
        
        // Test Comparison (includes AM and DDAM)
        NSLog(@"=== Method Comparison (AM & DDAM) ===");
        DDRKAMComparisonResults* compResults = [DDRKAMComparison compareMethodsWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t; (void)params;
            dydt[0] = -y[0];
        } startTime:t0 endTime:tEnd initialState:y0 stepSize:h exactSolution:exact params:NULL];
        
        if (compResults) {
            NSLog(@"  AM Accuracy: %.6f%%", [compResults.adamsAccuracy doubleValue] * 100.0);
            NSLog(@"  DDAM Accuracy: %.6f%%", [compResults.ddamAccuracy doubleValue] * 100.0);
            NSLog(@"  ✅ Comparison complete\n");
        }
        
        NSLog(@"=== Exponential Decay Test Complete ===\n");
    }
}

void testHarmonicOscillatorObjectiveC() {
    @autoreleasepool {
        NSLog(@"╔══════════════════════════════════════════════════════════════╗");
        NSLog(@"║  Objective-C: Harmonic Oscillator Test - All Methods          ║");
        NSLog(@"╚══════════════════════════════════════════════════════════════╝\n");
        
        NSLog(@"Test Case: d²x/dt² = -x, x(0) = 1.0, v(0) = 0.0, t ∈ [0, 2π]");
        NSLog(@"Exact Solution: x(t) = cos(t), v(t) = -sin(t)");
        double tEnd = 2.0 * M_PI;
        double xExact = cos(tEnd);
        double vExact = -sin(tEnd);
        NSLog(@"Expected Final Values: x(2π) = %.10f, v(2π) = %.10f\n", xExact, vExact);
        
        NSArray<NSNumber*>* y0 = @[@1.0, @0.0];
        NSArray<NSNumber*>* exact = @[@(xExact), @(vExact)];
        double t0 = 0.0;
        double h = 0.01;
        
        // Test RK3
        NSLog(@"=== RK3 Method ===");
        DDRKAMSolver* rk3Solver = [[DDRKAMSolver alloc] initWithDimension:2];
        NSDate* start = [NSDate date];
        NSDictionary* rk3Result = [rk3Solver solveWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t; (void)params;
            dydt[0] = y[1];   // dx/dt = v
            dydt[1] = -y[0];  // dv/dt = -x
        } startTime:t0 endTime:tEnd initialState:y0 stepSize:h params:NULL];
        NSTimeInterval rk3Time = [[NSDate date] timeIntervalSinceDate:start];
        
        if (rk3Result) {
            NSArray<NSArray<NSNumber*>*>* states = rk3Result[@"state"];
            NSArray<NSNumber*>* finalState = states.lastObject;
            double xFinal = [finalState[0] doubleValue];
            double vFinal = [finalState[1] doubleValue];
            double errorX = fabs(xFinal - xExact);
            double errorV = fabs(vFinal - vExact);
            double errorTotal = sqrt(errorX * errorX + errorV * errorV);
            double accuracy = (1.0 - errorTotal / sqrt(xExact * xExact + vExact * vExact)) * 100.0;
            NSLog(@"  Time: %.6f seconds", rk3Time);
            NSLog(@"  Final position: %.10f (exact: %.10f, error: %.6e)", xFinal, xExact, errorX);
            NSLog(@"  Final velocity: %.10f (exact: %.10f, error: %.6e)", vFinal, vExact, errorV);
            NSLog(@"  Total error: %.6e", errorTotal);
            NSLog(@"  Accuracy: %.6f%%", accuracy);
            NSLog(@"  %@\n", (errorTotal < 0.01) ? @"PASS" : @"FAIL");
        }
        
        // Test DDRK3
        NSLog(@"=== DDRK3 Method ===");
        DDRKAMHierarchicalSolver* ddrk3Solver = [[DDRKAMHierarchicalSolver alloc] 
                                                 initWithDimension:2 numLayers:3 hiddenDim:16];
        start = [NSDate date];
        NSDictionary* ddrk3Result = [ddrk3Solver solveWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t; (void)params;
            dydt[0] = y[1];
            dydt[1] = -y[0];
        } startTime:t0 endTime:tEnd initialState:y0 stepSize:h params:NULL];
        NSTimeInterval ddrk3Time = [[NSDate date] timeIntervalSinceDate:start];
        
        if (ddrk3Result) {
            NSArray<NSArray<NSNumber*>*>* states = ddrk3Result[@"state"];
            NSArray<NSNumber*>* finalState = states.lastObject;
            double xFinal = [finalState[0] doubleValue];
            double vFinal = [finalState[1] doubleValue];
            double errorX = fabs(xFinal - xExact);
            double errorV = fabs(vFinal - vExact);
            double errorTotal = sqrt(errorX * errorX + errorV * errorV);
            double accuracy = (1.0 - errorTotal / sqrt(xExact * xExact + vExact * vExact)) * 100.0;
            NSLog(@"  Time: %.6f seconds", ddrk3Time);
            NSLog(@"  Final position: %.10f (error: %.6e)", xFinal, errorX);
            NSLog(@"  Final velocity: %.10f (error: %.6e)", vFinal, errorV);
            NSLog(@"  Total error: %.6e", errorTotal);
            NSLog(@"  Accuracy: %.6f%%", accuracy);
            NSLog(@"  %@\n", (errorTotal < 0.01) ? @"PASS" : @"FAIL");
        }
        
        // Test Comparison (includes AM and DDAM)
        NSLog(@"=== Method Comparison (AM & DDAM) ===");
        DDRKAMComparisonResults* compResults = [DDRKAMComparison compareMethodsWithFunction:^(double t, const double* y, double* dydt, void* params) {
            (void)t; (void)params;
            dydt[0] = y[1];
            dydt[1] = -y[0];
        } startTime:t0 endTime:tEnd initialState:y0 stepSize:h exactSolution:exact params:NULL];
        
        if (compResults) {
            NSLog(@"  AM Accuracy: %.6f%%", [compResults.adamsAccuracy doubleValue] * 100.0);
            NSLog(@"  DDAM Accuracy: %.6f%%", [compResults.ddamAccuracy doubleValue] * 100.0);
            NSLog(@"  ✅ Comparison complete\n");
        }
        
        NSLog(@"=== Harmonic Oscillator Test Complete ===\n");
    }
}

int main(int argc, const char * argv[]) {
    testExponentialDecayObjectiveC();
    testHarmonicOscillatorObjectiveC();
    NSLog(@"✅ All Objective-C tests complete");
    return 0;
}
