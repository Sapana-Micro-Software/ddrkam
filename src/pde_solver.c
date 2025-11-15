/*
 * PDE Solver Implementation
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "pde_solver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

int pde_problem_init(PDEProblem* problem, PDEType type, SpatialDimension dim,
                     size_t nx, size_t ny, size_t nz,
                     double dx, double dy, double dz, double dt) {
    if (!problem || nx == 0 || dx <= 0 || dt <= 0) {
        return -1;
    }
    
    problem->type = type;
    problem->dim = dim;
    problem->nx = nx;
    problem->ny = (dim >= DIM_2D) ? ny : 1;
    problem->nz = (dim == DIM_3D) ? nz : 1;
    problem->dx = dx;
    problem->dy = (dim >= DIM_2D) ? dy : 1.0;
    problem->dz = (dim == DIM_3D) ? dz : 1.0;
    problem->dt = dt;
    
    // Default coefficients
    problem->alpha = 0.1;  // Diffusion coefficient
    problem->c = 1.0;      // Wave speed
    problem->nu = 0.01;     // Viscosity
    problem->a = 1.0;       // Advection speed
    
    size_t n_points = problem->nx * problem->ny * problem->nz;
    problem->initial_condition = (double*)calloc(n_points, sizeof(double));
    problem->boundary_condition = (double*)calloc(n_points, sizeof(double));
    
    if (!problem->initial_condition || !problem->boundary_condition) {
        pde_problem_free(problem);
        return -1;
    }
    
    return 0;
}

void pde_problem_free(PDEProblem* problem) {
    if (!problem) return;
    
    if (problem->initial_condition) {
        free(problem->initial_condition);
        problem->initial_condition = NULL;
    }
    if (problem->boundary_condition) {
        free(problem->boundary_condition);
        problem->boundary_condition = NULL;
    }
}

int pde_solve_heat_1d(const PDEProblem* problem, double t_end, PDESolution* solution) {
    if (!problem || !solution || problem->dim != DIM_1D || problem->type != PDE_HEAT) {
        return -1;
    }
    
    size_t nx = problem->nx;
    double dx = problem->dx;
    double dt = problem->dt;
    double alpha = problem->alpha;
    double r = alpha * dt / (dx * dx);
    
    // Stability condition: r <= 0.5
    if (r > 0.5) {
        return -1;
    }
    
    size_t n_steps = (size_t)(t_end / dt) + 1;
    solution->n_points = nx;
    solution->n_time_steps = n_steps;
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(nx * sizeof(double));
    solution->u_prev = (double*)malloc(nx * sizeof(double));
    solution->time = (double*)malloc(n_steps * sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize from initial condition
    if (problem->initial_condition) {
        memcpy(solution->u, problem->initial_condition, nx * sizeof(double));
    } else {
        // Default: Gaussian initial condition
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            solution->u[i] = exp(-(x - 0.5) * (x - 0.5) / 0.01);
        }
    }
    
    // Apply boundary conditions
    if (problem->boundary_condition) {
        solution->u[0] = problem->boundary_condition[0];
        solution->u[nx-1] = problem->boundary_condition[nx-1];
    } else {
        // Default: Dirichlet boundary conditions (zero)
        solution->u[0] = 0.0;
        solution->u[nx-1] = 0.0;
    }
    
    solution->time[0] = 0.0;
    
    // Time stepping using explicit finite difference
    for (size_t n = 1; n < n_steps; n++) {
        memcpy(solution->u_prev, solution->u, nx * sizeof(double));
        
        // Interior points: u_i^(n+1) = u_i^n + r * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
        for (size_t i = 1; i < nx - 1; i++) {
            solution->u[i] = solution->u_prev[i] + 
                            r * (solution->u_prev[i+1] - 2.0 * solution->u_prev[i] + solution->u_prev[i-1]);
        }
        
        // Boundary conditions
        solution->u[0] = (problem->boundary_condition) ? problem->boundary_condition[0] : 0.0;
        solution->u[nx-1] = (problem->boundary_condition) ? problem->boundary_condition[nx-1] : 0.0;
        
        solution->current_time = n * dt;
        solution->time[n] = solution->current_time;
    }
    
    return 0;
}

int pde_solve_heat_2d(const PDEProblem* problem, double t_end, PDESolution* solution) {
    if (!problem || !solution || problem->dim != DIM_2D || problem->type != PDE_HEAT) {
        return -1;
    }
    
    size_t nx = problem->nx;
    size_t ny = problem->ny;
    double dx = problem->dx;
    double dy = problem->dy;
    double dt = problem->dt;
    double alpha = problem->alpha;
    double rx = alpha * dt / (dx * dx);
    double ry = alpha * dt / (dy * dy);
    
    // Stability condition
    if (rx + ry > 0.5) {
        return -1;
    }
    
    size_t n_steps = (size_t)(t_end / dt) + 1;
    size_t n_points = nx * ny;
    solution->n_points = n_points;
    solution->n_time_steps = n_steps;
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(n_points * sizeof(double));
    solution->u_prev = (double*)malloc(n_points * sizeof(double));
    solution->time = (double*)malloc(n_steps * sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize
    if (problem->initial_condition) {
        memcpy(solution->u, problem->initial_condition, n_points * sizeof(double));
    } else {
        // Default: 2D Gaussian
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                double x = i * dx;
                double y = j * dy;
                solution->u[j * nx + i] = exp(-((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) / 0.01);
            }
        }
    }
    
    solution->time[0] = 0.0;
    
    // Time stepping: 2D heat equation
    for (size_t n = 1; n < n_steps; n++) {
        memcpy(solution->u_prev, solution->u, n_points * sizeof(double));
        
        // Interior points
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                double laplacian = (solution->u_prev[idx + 1] - 2.0 * solution->u_prev[idx] + solution->u_prev[idx - 1]) / (dx * dx) +
                                  (solution->u_prev[idx + nx] - 2.0 * solution->u_prev[idx] + solution->u_prev[idx - nx]) / (dy * dy);
                solution->u[idx] = solution->u_prev[idx] + alpha * dt * laplacian;
            }
        }
        
        // Boundary conditions (zero for simplicity)
        for (size_t i = 0; i < nx; i++) {
            solution->u[i] = 0.0;  // Bottom
            solution->u[(ny-1) * nx + i] = 0.0;  // Top
        }
        for (size_t j = 0; j < ny; j++) {
            solution->u[j * nx] = 0.0;  // Left
            solution->u[j * nx + nx - 1] = 0.0;  // Right
        }
        
        solution->current_time = n * dt;
        solution->time[n] = solution->current_time;
    }
    
    return 0;
}

int pde_solve_wave_1d(const PDEProblem* problem, double t_end, PDESolution* solution) {
    if (!problem || !solution || problem->dim != DIM_1D || problem->type != PDE_WAVE) {
        return -1;
    }
    
    size_t nx = problem->nx;
    double dx = problem->dx;
    double dt = problem->dt;
    double c = problem->c;
    double r = c * dt / dx;
    
    // Stability condition: r <= 1 (CFL condition)
    if (r > 1.0) {
        return -1;
    }
    
    size_t n_steps = (size_t)(t_end / dt) + 1;
    solution->n_points = nx;
    solution->n_time_steps = n_steps;
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(nx * sizeof(double));
    solution->u_prev = (double*)malloc(nx * sizeof(double));
    solution->time = (double*)malloc(n_steps * sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize
    if (problem->initial_condition) {
        memcpy(solution->u, problem->initial_condition, nx * sizeof(double));
    } else {
        // Default: Sine wave
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            solution->u[i] = sin(M_PI * x);
        }
    }
    
    // Initial velocity (zero for simplicity)
    double* u_vel = (double*)calloc(nx, sizeof(double));
    if (!u_vel) {
        pde_solution_free(solution);
        return -1;
    }
    
    solution->time[0] = 0.0;
    
    // First time step (using initial condition)
    memcpy(solution->u_prev, solution->u, nx * sizeof(double));
    
    // Time stepping: u_tt = c² * u_xx
    for (size_t n = 1; n < n_steps; n++) {
        // Interior points
        for (size_t i = 1; i < nx - 1; i++) {
            double uxx = (solution->u[i+1] - 2.0 * solution->u[i] + solution->u[i-1]) / (dx * dx);
            double u_new = 2.0 * solution->u[i] - solution->u_prev[i] + c * c * dt * dt * uxx;
            solution->u_prev[i] = solution->u[i];
            solution->u[i] = u_new;
        }
        
        // Boundary conditions
        solution->u[0] = 0.0;
        solution->u[nx-1] = 0.0;
        
        solution->current_time = n * dt;
        solution->time[n] = solution->current_time;
    }
    
    free(u_vel);
    return 0;
}

int pde_solve_advection_1d(const PDEProblem* problem, double t_end, PDESolution* solution) {
    if (!problem || !solution || problem->dim != DIM_1D || problem->type != PDE_ADVECTION) {
        return -1;
    }
    
    size_t nx = problem->nx;
    double dx = problem->dx;
    double dt = problem->dt;
    double a = problem->a;
    double r = a * dt / dx;
    
    // Stability condition: |r| <= 1
    if (fabs(r) > 1.0) {
        return -1;
    }
    
    size_t n_steps = (size_t)(t_end / dt) + 1;
    solution->n_points = nx;
    solution->n_time_steps = n_steps;
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(nx * sizeof(double));
    solution->u_prev = (double*)malloc(nx * sizeof(double));
    solution->time = (double*)malloc(n_steps * sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize
    if (problem->initial_condition) {
        memcpy(solution->u, problem->initial_condition, nx * sizeof(double));
    } else {
        // Default: Gaussian pulse
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            solution->u[i] = exp(-(x - 0.3) * (x - 0.3) / 0.01);
        }
    }
    
    solution->time[0] = 0.0;
    
    // Time stepping using upwind scheme
    for (size_t n = 1; n < n_steps; n++) {
        memcpy(solution->u_prev, solution->u, nx * sizeof(double));
        
        if (a > 0) {
            // Upwind: u_i^(n+1) = u_i^n - r * (u_i^n - u_{i-1}^n)
            for (size_t i = 1; i < nx; i++) {
                solution->u[i] = solution->u_prev[i] - r * (solution->u_prev[i] - solution->u_prev[i-1]);
            }
        } else {
            // Upwind for negative a
            for (size_t i = 0; i < nx - 1; i++) {
                solution->u[i] = solution->u_prev[i] - r * (solution->u_prev[i+1] - solution->u_prev[i]);
            }
        }
        
        solution->current_time = n * dt;
        solution->time[n] = solution->current_time;
    }
    
    return 0;
}

int pde_solve_burgers_1d(const PDEProblem* problem, double t_end, PDESolution* solution) {
    if (!problem || !solution || problem->dim != DIM_1D || problem->type != PDE_BURGERS) {
        return -1;
    }
    
    size_t nx = problem->nx;
    double dx = problem->dx;
    double dt = problem->dt;
    double nu = problem->nu;
    
    size_t n_steps = (size_t)(t_end / dt) + 1;
    solution->n_points = nx;
    solution->n_time_steps = n_steps;
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(nx * sizeof(double));
    solution->u_prev = (double*)malloc(nx * sizeof(double));
    solution->time = (double*)malloc(n_steps * sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize
    if (problem->initial_condition) {
        memcpy(solution->u, problem->initial_condition, nx * sizeof(double));
    } else {
        // Default: Step function
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            solution->u[i] = (x < 0.5) ? 1.0 : 0.0;
        }
    }
    
    solution->time[0] = 0.0;
    
    // Time stepping: u_t + u * u_x = ν * u_xx
    for (size_t n = 1; n < n_steps; n++) {
        memcpy(solution->u_prev, solution->u, nx * sizeof(double));
        
        // Interior points using finite difference
        for (size_t i = 1; i < nx - 1; i++) {
            double ux = (solution->u_prev[i+1] - solution->u_prev[i-1]) / (2.0 * dx);
            double uxx = (solution->u_prev[i+1] - 2.0 * solution->u_prev[i] + solution->u_prev[i-1]) / (dx * dx);
            solution->u[i] = solution->u_prev[i] - dt * solution->u_prev[i] * ux + nu * dt * uxx;
        }
        
        // Boundary conditions
        solution->u[0] = solution->u_prev[0];
        solution->u[nx-1] = solution->u_prev[nx-1];
        
        solution->current_time = n * dt;
        solution->time[n] = solution->current_time;
    }
    
    return 0;
}

int pde_solve_laplace_2d(const PDEProblem* problem, PDESolution* solution) {
    if (!problem || !solution || problem->dim != DIM_2D || problem->type != PDE_LAPLACE) {
        return -1;
    }
    
    size_t nx = problem->nx;
    size_t ny = problem->ny;
    double dx = problem->dx;
    double dy = problem->dy;
    size_t n_points = nx * ny;
    
    solution->n_points = n_points;
    solution->n_time_steps = 1;  // Steady-state problem
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(n_points * sizeof(double));
    solution->u_prev = (double*)malloc(n_points * sizeof(double));
    solution->time = (double*)malloc(sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize
    if (problem->initial_condition) {
        memcpy(solution->u, problem->initial_condition, n_points * sizeof(double));
    } else {
        memset(solution->u, 0, n_points * sizeof(double));
    }
    
    // Apply boundary conditions
    if (problem->boundary_condition) {
        for (size_t i = 0; i < nx; i++) {
            solution->u[i] = problem->boundary_condition[i];  // Bottom
            solution->u[(ny-1) * nx + i] = problem->boundary_condition[(ny-1) * nx + i];  // Top
        }
        for (size_t j = 0; j < ny; j++) {
            solution->u[j * nx] = problem->boundary_condition[j * nx];  // Left
            solution->u[j * nx + nx - 1] = problem->boundary_condition[j * nx + nx - 1];  // Right
        }
    }
    
    // Iterative solution using Jacobi method
    double tolerance = 1e-6;
    size_t max_iter = 10000;
    
    for (size_t iter = 0; iter < max_iter; iter++) {
        memcpy(solution->u_prev, solution->u, n_points * sizeof(double));
        double max_diff = 0.0;
        
        // Interior points: u_xx + u_yy = 0
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                
                // Jacobi update (simplified)
                double beta = 2.0 * (1.0/(dx*dx) + 1.0/(dy*dy));
                solution->u[idx] = ((solution->u_prev[idx+1] + solution->u_prev[idx-1]) / (dx*dx) +
                                   (solution->u_prev[idx+nx] + solution->u_prev[idx-nx]) / (dy*dy)) / beta;
                
                double diff = fabs(solution->u[idx] - solution->u_prev[idx]);
                if (diff > max_diff) max_diff = diff;
            }
        }
        
        if (max_diff < tolerance) {
            break;
        }
    }
    
    solution->time[0] = 0.0;
    return 0;
}

int pde_solve_poisson_2d(const PDEProblem* problem, 
                        double (*f)(double x, double y, void* params),
                        void* params, PDESolution* solution) {
    if (!problem || !solution || !f || problem->dim != DIM_2D || problem->type != PDE_POISSON) {
        return -1;
    }
    
    size_t nx = problem->nx;
    size_t ny = problem->ny;
    double dx = problem->dx;
    double dy = problem->dy;
    size_t n_points = nx * ny;
    
    solution->n_points = n_points;
    solution->n_time_steps = 1;
    solution->current_time = 0.0;
    
    solution->u = (double*)malloc(n_points * sizeof(double));
    solution->u_prev = (double*)malloc(n_points * sizeof(double));
    solution->time = (double*)malloc(sizeof(double));
    
    if (!solution->u || !solution->u_prev || !solution->time) {
        pde_solution_free(solution);
        return -1;
    }
    
    // Initialize
    memset(solution->u, 0, n_points * sizeof(double));
    
    // Apply boundary conditions
    if (problem->boundary_condition) {
        for (size_t i = 0; i < nx; i++) {
            solution->u[i] = problem->boundary_condition[i];
            solution->u[(ny-1) * nx + i] = problem->boundary_condition[(ny-1) * nx + i];
        }
        for (size_t j = 0; j < ny; j++) {
            solution->u[j * nx] = problem->boundary_condition[j * nx];
            solution->u[j * nx + nx - 1] = problem->boundary_condition[j * nx + nx - 1];
        }
    }
    
    // Iterative solution
    double tolerance = 1e-6;
    size_t max_iter = 10000;
    
    for (size_t iter = 0; iter < max_iter; iter++) {
        memcpy(solution->u_prev, solution->u, n_points * sizeof(double));
        double max_diff = 0.0;
        
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                double x = i * dx;
                double y = j * dy;
                double f_val = f(x, y, params);
                
                double beta = 2.0 * (1.0/(dx*dx) + 1.0/(dy*dy));
                solution->u[idx] = ((solution->u_prev[idx+1] + solution->u_prev[idx-1]) / (dx*dx) +
                                   (solution->u_prev[idx+nx] + solution->u_prev[idx-nx]) / (dy*dy) - f_val) / beta;
                
                double diff = fabs(solution->u[idx] - solution->u_prev[idx]);
                if (diff > max_diff) max_diff = diff;
            }
        }
        
        if (max_diff < tolerance) {
            break;
        }
    }
    
    solution->time[0] = 0.0;
    return 0;
}

void pde_solution_free(PDESolution* solution) {
    if (!solution) return;
    
    if (solution->u) {
        free(solution->u);
        solution->u = NULL;
    }
    if (solution->u_prev) {
        free(solution->u_prev);
        solution->u_prev = NULL;
    }
    if (solution->time) {
        free(solution->time);
        solution->time = NULL;
    }
}

int pde_export_solution(const PDESolution* solution, const PDEProblem* problem,
                       const char* filename) {
    if (!solution || !problem || !filename) {
        return -1;
    }
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        return -1;
    }
    
    if (problem->dim == DIM_1D) {
        fprintf(fp, "x,time,u\n");
        for (size_t n = 0; n < solution->n_time_steps; n++) {
            for (size_t i = 0; i < problem->nx; i++) {
                double x = i * problem->dx;
                double t = (n < solution->n_time_steps) ? solution->time[n] : 0.0;
                // For simplicity, export final solution
                if (n == solution->n_time_steps - 1) {
                    fprintf(fp, "%.6f,%.6f,%.6f\n", x, t, solution->u[i]);
                }
            }
        }
    } else if (problem->dim == DIM_2D) {
        fprintf(fp, "x,y,time,u\n");
        for (size_t j = 0; j < problem->ny; j++) {
            for (size_t i = 0; i < problem->nx; i++) {
                double x = i * problem->dx;
                double y = j * problem->dy;
                size_t idx = j * problem->nx + i;
                double t = (solution->n_time_steps > 0) ? solution->time[solution->n_time_steps - 1] : 0.0;
                fprintf(fp, "%.6f,%.6f,%.6f,%.6f\n", x, y, t, solution->u[idx]);
            }
        }
    }
    
    fclose(fp);
    return 0;
}
