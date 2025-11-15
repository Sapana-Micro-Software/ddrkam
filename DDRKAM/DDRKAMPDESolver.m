/*
 * DDRKAM PDE Solver Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#import "DDRKAMPDESolver.h"
#import "../include/pde_solver.h"

@implementation DDRKAMPDESolution

- (NSString*)description {
    return [NSString stringWithFormat:@"PDE Solution: %lu points, %lu time steps, t=%.6f",
            (unsigned long)self.gridPoints, (unsigned long)self.timeStepsCount, self.currentTime];
}

@end

@implementation DDRKAMPDESolver

+ (DDRKAMPDESolution*)solveHeat1DWithGridPoints:(NSUInteger)nx
                                       spatialStep:(double)dx
                                         timeStep:(double)dt
                                    diffusionCoeff:(double)alpha
                                  initialCondition:(NSArray<NSNumber*>*)u0
                                         finalTime:(double)tEnd {
    if (nx == 0 || dx <= 0 || dt <= 0 || !u0 || u0.count != nx) {
        return nil;
    }
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_HEAT, DIM_1D, nx, 1, 1, dx, 1.0, 1.0, dt) != 0) {
        return nil;
    }
    
    problem.alpha = alpha;
    
    for (NSUInteger i = 0; i < nx; i++) {
        problem.initial_condition[i] = [u0[i] doubleValue];
    }
    
    PDESolution solution;
    if (pde_solve_heat_1d(&problem, tEnd, &solution) != 0) {
        pde_problem_free(&problem);
        return nil;
    }
    
    DDRKAMPDESolution* result = [[DDRKAMPDESolution alloc] init];
    NSMutableArray<NSNumber*>* solArray = [NSMutableArray arrayWithCapacity:nx];
    for (NSUInteger i = 0; i < nx; i++) {
        [solArray addObject:@(solution.u[i])];
    }
    
    NSMutableArray<NSNumber*>* timeArray = [NSMutableArray arrayWithCapacity:solution.n_time_steps];
    for (size_t i = 0; i < solution.n_time_steps; i++) {
        [timeArray addObject:@(solution.time[i])];
    }
    
    [result setValue:solArray forKey:@"solution"];
    [result setValue:timeArray forKey:@"timeSteps"];
    [result setValue:@(solution.n_points) forKey:@"gridPoints"];
    [result setValue:@(solution.n_time_steps) forKey:@"timeStepsCount"];
    [result setValue:@(solution.current_time) forKey:@"currentTime"];
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    
    return result;
}

+ (DDRKAMPDESolution*)solveHeat2DWithGridPointsX:(NSUInteger)nx
                                            gridY:(NSUInteger)ny
                                       spatialStepX:(double)dx
                                       spatialStepY:(double)dy
                                         timeStep:(double)dt
                                    diffusionCoeff:(double)alpha
                                  initialCondition:(NSArray<NSArray<NSNumber*>*>*)u0
                                         finalTime:(double)tEnd {
    if (nx == 0 || ny == 0 || dx <= 0 || dy <= 0 || dt <= 0 || !u0 || u0.count != ny) {
        return nil;
    }
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_HEAT, DIM_2D, nx, ny, 1, dx, dy, 1.0, dt) != 0) {
        return nil;
    }
    
    problem.alpha = alpha;
    
    for (NSUInteger j = 0; j < ny; j++) {
        if (u0[j].count != nx) {
            pde_problem_free(&problem);
            return nil;
        }
        for (NSUInteger i = 0; i < nx; i++) {
            problem.initial_condition[j * nx + i] = [u0[j][i] doubleValue];
        }
    }
    
    PDESolution solution;
    if (pde_solve_heat_2d(&problem, tEnd, &solution) != 0) {
        pde_problem_free(&problem);
        return nil;
    }
    
    DDRKAMPDESolution* result = [[DDRKAMPDESolution alloc] init];
    NSMutableArray<NSNumber*>* solArray = [NSMutableArray arrayWithCapacity:nx * ny];
    for (NSUInteger i = 0; i < nx * ny; i++) {
        [solArray addObject:@(solution.u[i])];
    }
    
    NSMutableArray<NSNumber*>* timeArray = [NSMutableArray arrayWithCapacity:solution.n_time_steps];
    for (size_t i = 0; i < solution.n_time_steps; i++) {
        [timeArray addObject:@(solution.time[i])];
    }
    
    [result setValue:solArray forKey:@"solution"];
    [result setValue:timeArray forKey:@"timeSteps"];
    [result setValue:@(solution.n_points) forKey:@"gridPoints"];
    [result setValue:@(solution.n_time_steps) forKey:@"timeStepsCount"];
    [result setValue:@(solution.current_time) forKey:@"currentTime"];
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    
    return result;
}

+ (DDRKAMPDESolution*)solveWave1DWithGridPoints:(NSUInteger)nx
                                       spatialStep:(double)dx
                                         timeStep:(double)dt
                                        waveSpeed:(double)c
                                  initialCondition:(NSArray<NSNumber*>*)u0
                                         finalTime:(double)tEnd {
    if (nx == 0 || dx <= 0 || dt <= 0 || !u0 || u0.count != nx) {
        return nil;
    }
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_WAVE, DIM_1D, nx, 1, 1, dx, 1.0, 1.0, dt) != 0) {
        return nil;
    }
    
    problem.c = c;
    
    for (NSUInteger i = 0; i < nx; i++) {
        problem.initial_condition[i] = [u0[i] doubleValue];
    }
    
    PDESolution solution;
    if (pde_solve_wave_1d(&problem, tEnd, &solution) != 0) {
        pde_problem_free(&problem);
        return nil;
    }
    
    DDRKAMPDESolution* result = [[DDRKAMPDESolution alloc] init];
    NSMutableArray<NSNumber*>* solArray = [NSMutableArray arrayWithCapacity:nx];
    for (NSUInteger i = 0; i < nx; i++) {
        [solArray addObject:@(solution.u[i])];
    }
    
    NSMutableArray<NSNumber*>* timeArray = [NSMutableArray arrayWithCapacity:solution.n_time_steps];
    for (size_t i = 0; i < solution.n_time_steps; i++) {
        [timeArray addObject:@(solution.time[i])];
    }
    
    [result setValue:solArray forKey:@"solution"];
    [result setValue:timeArray forKey:@"timeSteps"];
    [result setValue:@(solution.n_points) forKey:@"gridPoints"];
    [result setValue:@(solution.n_time_steps) forKey:@"timeStepsCount"];
    [result setValue:@(solution.current_time) forKey:@"currentTime"];
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    
    return result;
}

+ (DDRKAMPDESolution*)solveAdvection1DWithGridPoints:(NSUInteger)nx
                                            spatialStep:(double)dx
                                              timeStep:(double)dt
                                          advectionSpeed:(double)a
                                       initialCondition:(NSArray<NSNumber*>*)u0
                                              finalTime:(double)tEnd {
    if (nx == 0 || dx <= 0 || dt <= 0 || !u0 || u0.count != nx) {
        return nil;
    }
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_ADVECTION, DIM_1D, nx, 1, 1, dx, 1.0, 1.0, dt) != 0) {
        return nil;
    }
    
    problem.a = a;
    
    for (NSUInteger i = 0; i < nx; i++) {
        problem.initial_condition[i] = [u0[i] doubleValue];
    }
    
    PDESolution solution;
    if (pde_solve_advection_1d(&problem, tEnd, &solution) != 0) {
        pde_problem_free(&problem);
        return nil;
    }
    
    DDRKAMPDESolution* result = [[DDRKAMPDESolution alloc] init];
    NSMutableArray<NSNumber*>* solArray = [NSMutableArray arrayWithCapacity:nx];
    for (NSUInteger i = 0; i < nx; i++) {
        [solArray addObject:@(solution.u[i])];
    }
    
    NSMutableArray<NSNumber*>* timeArray = [NSMutableArray arrayWithCapacity:solution.n_time_steps];
    for (size_t i = 0; i < solution.n_time_steps; i++) {
        [timeArray addObject:@(solution.time[i])];
    }
    
    [result setValue:solArray forKey:@"solution"];
    [result setValue:timeArray forKey:@"timeSteps"];
    [result setValue:@(solution.n_points) forKey:@"gridPoints"];
    [result setValue:@(solution.n_time_steps) forKey:@"timeStepsCount"];
    [result setValue:@(solution.current_time) forKey:@"currentTime"];
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    
    return result;
}

+ (BOOL)exportSolution:(DDRKAMPDESolution*)solution
                toFile:(NSString*)filePath
             dimension:(DDRKAMSpatialDimension)dim
             gridSizeX:(NSUInteger)nx
             gridSizeY:(NSUInteger)ny {
    if (!solution || !filePath) {
        return NO;
    }
    
    NSMutableString* csv = [NSMutableString string];
    
    if (dim == DDRKAMSpatialDimension1D) {
        [csv appendString:@"x,time,u\n"];
        for (NSUInteger i = 0; i < nx && i < solution.solution.count; i++) {
            double x = i * 0.01; // Assuming dx = 0.01
            double t = solution.currentTime;
            [csv appendFormat:@"%.6f,%.6f,%.6f\n", x, t, [solution.solution[i] doubleValue]];
        }
    } else if (dim == DDRKAMSpatialDimension2D) {
        [csv appendString:@"x,y,time,u\n"];
        for (NSUInteger j = 0; j < ny; j++) {
            for (NSUInteger i = 0; i < nx; i++) {
                NSUInteger idx = j * nx + i;
                if (idx < solution.solution.count) {
                    double x = i * 0.02;
                    double y = j * 0.02;
                    double t = solution.currentTime;
                    [csv appendFormat:@"%.6f,%.6f,%.6f,%.6f\n", x, y, t, [solution.solution[idx] doubleValue]];
                }
            }
        }
    }
    
    NSError* error = nil;
    BOOL success = [csv writeToFile:filePath atomically:YES encoding:NSUTF8StringEncoding error:&error];
    return success;
}

@end
