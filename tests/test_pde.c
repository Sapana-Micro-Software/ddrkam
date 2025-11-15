/*
 * PDE Solver Test Suite
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/pde_solver.h"

int test_heat_1d() {
    printf("Testing 1D Heat Equation...\n");
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_HEAT, DIM_1D, 100, 1, 1, 0.01, 1.0, 1.0, 0.0001) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    problem.alpha = 0.1;
    
    // Initial condition: Gaussian
    for (size_t i = 0; i < problem.nx; i++) {
        double x = i * problem.dx;
        problem.initial_condition[i] = exp(-(x - 0.5) * (x - 0.5) / 0.01);
    }
    
    PDESolution solution;
    if (pde_solve_heat_1d(&problem, 0.1, &solution) != 0) {
        printf("  FAIL: Solution failed\n");
        pde_problem_free(&problem);
        return 1;
    }
    
    printf("  Steps: %zu, Final time: %.6f\n", solution.n_time_steps, solution.current_time);
    printf("  Solution range: [%.6f, %.6f]\n", 
           solution.u[0], solution.u[problem.nx-1]);
    
    pde_export_solution(&solution, &problem, "heat_1d_solution.csv");
    printf("  ✅ Exported to heat_1d_solution.csv\n");
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    printf("  PASS\n\n");
    return 0;
}

int test_wave_1d() {
    printf("Testing 1D Wave Equation...\n");
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_WAVE, DIM_1D, 100, 1, 1, 0.01, 1.0, 1.0, 0.001) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    problem.c = 1.0;
    
    // Initial condition: Sine wave
    for (size_t i = 0; i < problem.nx; i++) {
        double x = i * problem.dx;
        problem.initial_condition[i] = sin(M_PI * x);
    }
    
    PDESolution solution;
    if (pde_solve_wave_1d(&problem, 0.5, &solution) != 0) {
        printf("  FAIL: Solution failed\n");
        pde_problem_free(&problem);
        return 1;
    }
    
    printf("  Steps: %zu, Final time: %.6f\n", solution.n_time_steps, solution.current_time);
    
    pde_export_solution(&solution, &problem, "wave_1d_solution.csv");
    printf("  ✅ Exported to wave_1d_solution.csv\n");
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    printf("  PASS\n\n");
    return 0;
}

int test_advection_1d() {
    printf("Testing 1D Advection Equation...\n");
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_ADVECTION, DIM_1D, 100, 1, 1, 0.01, 1.0, 1.0, 0.001) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    problem.a = 1.0;
    
    // Initial condition: Gaussian pulse
    for (size_t i = 0; i < problem.nx; i++) {
        double x = i * problem.dx;
        problem.initial_condition[i] = exp(-(x - 0.3) * (x - 0.3) / 0.01);
    }
    
    PDESolution solution;
    if (pde_solve_advection_1d(&problem, 0.2, &solution) != 0) {
        printf("  FAIL: Solution failed\n");
        pde_problem_free(&problem);
        return 1;
    }
    
    printf("  Steps: %zu, Final time: %.6f\n", solution.n_time_steps, solution.current_time);
    
    pde_export_solution(&solution, &problem, "advection_1d_solution.csv");
    printf("  ✅ Exported to advection_1d_solution.csv\n");
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    printf("  PASS\n\n");
    return 0;
}

int test_heat_2d() {
    printf("Testing 2D Heat Equation...\n");
    
    PDEProblem problem;
    if (pde_problem_init(&problem, PDE_HEAT, DIM_2D, 50, 50, 1, 0.02, 0.02, 1.0, 0.0001) != 0) {
        printf("  FAIL: Initialization failed\n");
        return 1;
    }
    
    problem.alpha = 0.1;
    
    // Initial condition: 2D Gaussian
    for (size_t j = 0; j < problem.ny; j++) {
        for (size_t i = 0; i < problem.nx; i++) {
            double x = i * problem.dx;
            double y = j * problem.dy;
            problem.initial_condition[j * problem.nx + i] = 
                exp(-((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) / 0.01);
        }
    }
    
    PDESolution solution;
    if (pde_solve_heat_2d(&problem, 0.05, &solution) != 0) {
        printf("  FAIL: Solution failed\n");
        pde_problem_free(&problem);
        return 1;
    }
    
    printf("  Steps: %zu, Final time: %.6f\n", solution.n_time_steps, solution.current_time);
    printf("  Grid: %zux%zu = %zu points\n", problem.nx, problem.ny, solution.n_points);
    
    pde_export_solution(&solution, &problem, "heat_2d_solution.csv");
    printf("  ✅ Exported to heat_2d_solution.csv\n");
    
    pde_solution_free(&solution);
    pde_problem_free(&problem);
    printf("  PASS\n\n");
    return 0;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          PDE Solver Test Suite (ODEs & PDEs)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int failures = 0;
    failures += test_heat_1d();
    failures += test_wave_1d();
    failures += test_advection_1d();
    failures += test_heat_2d();
    
    printf("=== Test Summary ===\n");
    printf("Failures: %d\n", failures);
    
    return (failures == 0) ? 0 : 1;
}
