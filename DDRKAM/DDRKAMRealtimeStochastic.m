/*
 * DDRKAM Real-Time and Stochastic Solvers Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMRealtimeStochastic.h"
#import "../include/realtime_stochastic.h"

@implementation DDRKAMStochasticParams

+ (instancetype)defaultParams {
    DDRKAMStochasticParams* params = [[DDRKAMStochasticParams alloc] init];
    params.noiseAmplitude = 0.01;
    params.noiseCorrelation = 0.1;
    params.useBrownianMotion = NO;
    params.randomSeed = 0.0;
    return params;
}

+ (instancetype)paramsWithAmplitude:(double)amplitude useBrownian:(BOOL)useBrownian {
    DDRKAMStochasticParams* params = [[DDRKAMStochasticParams alloc] init];
    params.noiseAmplitude = amplitude;
    params.noiseCorrelation = 0.1;
    params.useBrownianMotion = useBrownian;
    params.randomSeed = 0.0;
    return params;
}

@end

@implementation DDRKAMRealtimeSolver {
    RealtimeSolverState _state;
    void* _solverPtr;
}

- (instancetype)initWithDimension:(NSUInteger)n
                         stepSize:(double)h
                       bufferSize:(NSUInteger)bufferSize {
    self = [super init];
    if (self) {
        if (realtime_rk3_init(&_state, n, h, bufferSize) != 0) {
            return nil;
        }
        _solverPtr = &_state;
    }
    return self;
}

- (void)dealloc {
    realtime_solver_free(&_state);
}

- (NSUInteger)dimension {
    return _state.n;
}

- (double)stepSize {
    return _state.h;
}

- (NSUInteger)totalSteps {
    return (NSUInteger)_state.step_count;
}

- (double)currentTime {
    return _state.t_current;
}

- (BOOL)stepRK3WithFunction:(void (^)(double, const double*, double*, void*))f
                    newState:(NSArray<NSNumber*>*)yNew
                      params:(void*)params
                    callback:(void (^)(double, NSArray<NSNumber*>*))callback {
    if (yNew.count != self.dimension) {
        return NO;
    }
    
    // Convert NSArray to C array
    double* yArray = (double*)malloc(self.dimension * sizeof(double));
    if (!yArray) return NO;
    
    for (NSUInteger i = 0; i < self.dimension; i++) {
        yArray[i] = [yNew[i] doubleValue];
    }
    
    // Create C callback wrapper
    RealtimeCallback cCallback = NULL;
    void* userData = NULL;
    
    if (callback) {
        cCallback = ^(double t, const double* y, size_t n, void* data) {
            NSMutableArray<NSNumber*>* yArray = [NSMutableArray arrayWithCapacity:n];
            for (size_t i = 0; i < n; i++) {
                [yArray addObject:@(y[i])];
            }
            void (^block)(double, NSArray<NSNumber*>*) = (__bridge void (^)(double, NSArray<NSNumber*>*))data;
            block(t, yArray);
        };
        userData = (__bridge void*)callback;
    }
    
    // Create ODE function wrapper
    void (^odeBlock)(double, const double*, double*, void*) = f;
    void* odeBlockPtr = (__bridge void*)odeBlock;
    
    void odeFunction(double t, const double* y, double* dydt, void* p) {
        void (^block)(double, const double*, double*, void*) = (__bridge void (^)(double, const double*, double*, void*))odeBlockPtr;
        block(t, y, dydt, p);
    }
    
    int result = realtime_rk3_step(&_state, odeFunction, yArray, params, cCallback, userData);
    
    free(yArray);
    return (result == 0);
}

- (BOOL)stepAdamsWithFunction:(void (^)(double, const double*, double*, void*))f
                      newState:(NSArray<NSNumber*>*)yNew
                        params:(void*)params
                      callback:(void (^)(double, NSArray<NSNumber*>*))callback {
    if (yNew.count != self.dimension) {
        return NO;
    }
    
    double* yArray = (double*)malloc(self.dimension * sizeof(double));
    if (!yArray) return NO;
    
    for (NSUInteger i = 0; i < self.dimension; i++) {
        yArray[i] = [yNew[i] doubleValue];
    }
    
    RealtimeCallback cCallback = NULL;
    if (callback) {
        cCallback = ^(double t, const double* y, size_t n, void* data) {
            NSMutableArray<NSNumber*>* yArray = [NSMutableArray arrayWithCapacity:n];
            for (size_t i = 0; i < n; i++) {
                [yArray addObject:@(y[i])];
            }
            void (^block)(double, NSArray<NSNumber*>*) = (__bridge void (^)(double, NSArray<NSNumber*>*))data;
            block(t, yArray);
        };
    }
    
    void (^odeBlock)(double, const double*, double*, void*) = f;
    void* odeBlockPtr = (__bridge void*)odeBlock;
    
    void odeFunction(double t, const double* y, double* dydt, void* p) {
        void (^block)(double, const double*, double*, void*) = (__bridge void (^)(double, const double*, double*, void*))odeBlockPtr;
        block(t, y, dydt, p);
    }
    
    int result = realtime_adams_step(&_state, odeFunction, yArray, params, cCallback, (__bridge void*)callback);
    
    free(yArray);
    return (result == 0);
}

@end

@implementation DDRKAMStochasticSolver {
    void* _solver;
    BOOL _isRK3;
}

- (instancetype)initRK3WithDimension:(NSUInteger)n
                             stepSize:(double)h
                               params:(DDRKAMStochasticParams*)params {
    self = [super init];
    if (self) {
        StochasticParams cParams = {
            .noise_amplitude = params.noiseAmplitude,
            .noise_correlation = params.noiseCorrelation,
            .use_brownian = params.useBrownianMotion ? 1 : 0,
            .seed = params.randomSeed
        };
        
        _solver = stochastic_rk3_init(n, h, &cParams);
        if (!_solver) {
            return nil;
        }
        _isRK3 = YES;
    }
    return self;
}

- (instancetype)initAdamsWithDimension:(NSUInteger)n
                              stepSize:(double)h
                                params:(DDRKAMStochasticParams*)params {
    self = [super init];
    if (self) {
        StochasticParams cParams = {
            .noise_amplitude = params.noiseAmplitude,
            .noise_correlation = params.noiseCorrelation,
            .use_brownian = params.useBrownianMotion ? 1 : 0,
            .seed = params.randomSeed
        };
        
        _solver = stochastic_adams_init(n, h, &cParams);
        if (!_solver) {
            return nil;
        }
        _isRK3 = NO;
    }
    return self;
}

- (void)dealloc {
    if (_solver) {
        stochastic_solver_free(_solver);
    }
}

- (NSUInteger)dimension {
    return 0; // Would need to store this
}

- (double)stepSize {
    return 0.0; // Would need to store this
}

- (DDRKAMStochasticParams*)params {
    return nil; // Would need to store this
}

- (double)stepRK3WithFunction:(void (^)(double, const double*, double*, void*))f
                          time:(double)t0
                         state:(NSMutableArray<NSNumber*>*)y0
                        params:(void*)params {
    if (!_solver || !_isRK3) {
        return t0;
    }
    
    double* yArray = (double*)malloc(y0.count * sizeof(double));
    if (!yArray) return t0;
    
    for (NSUInteger i = 0; i < y0.count; i++) {
        yArray[i] = [y0[i] doubleValue];
    }
    
    void (^odeBlock)(double, const double*, double*, void*) = f;
    void* odeBlockPtr = (__bridge void*)odeBlock;
    
    void odeFunction(double t, const double* y, double* dydt, void* p) {
        void (^block)(double, const double*, double*, void*) = (__bridge void (^)(double, const double*, double*, void*))odeBlockPtr;
        block(t, y, dydt, p);
    }
    
    double tNew = stochastic_rk3_step(_solver, odeFunction, t0, yArray, params);
    
    for (NSUInteger i = 0; i < y0.count; i++) {
        y0[i] = @(yArray[i]);
    }
    
    free(yArray);
    return tNew;
}

- (double)stepAdamsWithFunction:(void (^)(double, const double*, double*, void*))f
                          time:(double)t0
                         state:(NSMutableArray<NSNumber*>*)y0
                        params:(void*)params {
    if (!_solver || _isRK3) {
        return t0;
    }
    
    double* yArray = (double*)malloc(y0.count * sizeof(double));
    if (!yArray) return t0;
    
    for (NSUInteger i = 0; i < y0.count; i++) {
        yArray[i] = [y0[i] doubleValue];
    }
    
    void (^odeBlock)(double, const double*, double*, void*) = f;
    void* odeBlockPtr = (__bridge void*)odeBlock;
    
    void odeFunction(double t, const double* y, double* dydt, void* p) {
        void (^block)(double, const double*, double*, void*) = (__bridge void (^)(double, const double*, double*, void*))odeBlockPtr;
        block(t, y, dydt, p);
    }
    
    double tNew = stochastic_adams_step(_solver, odeFunction, t0, yArray, params);
    
    for (NSUInteger i = 0; i < y0.count; i++) {
        y0[i] = @(yArray[i]);
    }
    
    free(yArray);
    return tNew;
}

@end

@implementation DDRKAMDataDrivenControl

+ (double)adaptiveStepSizeWithErrorHistory:(NSArray<NSNumber*>*)errorHistory
                                 currentH:(double)currentH
                              targetError:(double)targetError {
    if (errorHistory.count == 0) {
        return currentH;
    }
    
    double* errors = (double*)malloc(errorHistory.count * sizeof(double));
    if (!errors) return currentH;
    
    for (NSUInteger i = 0; i < errorHistory.count; i++) {
        errors[i] = [errorHistory[i] doubleValue];
    }
    
    double newH = data_driven_adaptive_step(errors, errorHistory.count, currentH, targetError);
    
    free(errors);
    return newH;
}

+ (NSInteger)selectMethodWithStiffness:(double)stiffness
                        errorTolerance:(double)tolerance
                       speedRequirement:(double)speed {
    int result = data_driven_method_select(stiffness, tolerance, speed);
    return (NSInteger)result;
}

@end
