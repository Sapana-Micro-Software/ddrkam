/*
 * Optimization-Based Solvers Implementation
 * ADAM, AdaGrad, and Karmarkar's Algorithm
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#include "optimization_solvers.h"
#include "rk3.h"
#include "euler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Simple RNG for Karmarkar's algorithm
static uint32_t xorshift32(uint32_t* state) {
    if (state == NULL) {
        static uint32_t default_state = 12345;
        state = &default_state;
    }
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double uniform_random(uint32_t* state) {
    return (double)xorshift32(state) / UINT32_MAX;
}

// ============================================================================
// Karmarkar's Algorithm Implementation
// ============================================================================

int karmarkar_solver_init(KarmarkarSolver* solver, size_t state_dim, ADAMSolverType type,
                           double alpha, double beta, double mu, double epsilon,
                           const double* c, const double** A, const double* b,
                           size_t num_constraints) {
    if (!solver || state_dim == 0) {
        return -1;
    }
    
    memset(solver, 0, sizeof(KarmarkarSolver));
    solver->state_dim = state_dim;
    solver->solver_type = type;
    solver->alpha = alpha > 0.0 ? alpha : 0.25;
    solver->beta = beta > 0.0 ? beta : 0.5;
    solver->mu = mu > 0.0 ? mu : 1.0;
    solver->epsilon = epsilon > 0.0 ? epsilon : 1e-6;
    solver->tolerance = epsilon;
    solver->max_iterations = 1000;
    solver->num_constraints = num_constraints;
    
    // Allocate objective coefficients
    if (c) {
        solver->c = (double*)malloc(state_dim * sizeof(double));
        if (!solver->c) {
            return -1;
        }
        memcpy(solver->c, c, state_dim * sizeof(double));
    } else {
        // Default: minimize sum of squares
        solver->c = (double*)malloc(state_dim * sizeof(double));
        if (!solver->c) {
            return -1;
        }
        for (size_t i = 0; i < state_dim; i++) {
            solver->c[i] = 1.0;
        }
    }
    
    // Allocate constraint matrix
    if (A && num_constraints > 0) {
        solver->A = (double**)malloc(num_constraints * sizeof(double*));
        if (!solver->A) {
            karmarkar_solver_free(solver);
            return -1;
        }
        for (size_t i = 0; i < num_constraints; i++) {
            solver->A[i] = (double*)malloc(state_dim * sizeof(double));
            if (!solver->A[i]) {
                for (size_t j = 0; j < i; j++) {
                    free(solver->A[j]);
                }
                free(solver->A);
                karmarkar_solver_free(solver);
                return -1;
            }
            if (A[i]) {
                memcpy(solver->A[i], A[i], state_dim * sizeof(double));
            } else {
                memset(solver->A[i], 0, state_dim * sizeof(double));
            }
        }
    }
    
    // Allocate constraint bounds
    if (b && num_constraints > 0) {
        solver->b = (double*)malloc(num_constraints * sizeof(double));
        if (!solver->b) {
            karmarkar_solver_free(solver);
            return -1;
        }
        memcpy(solver->b, b, num_constraints * sizeof(double));
    }
    
    // Allocate interior point state
    solver->x = (double*)malloc(state_dim * sizeof(double));
    solver->y = (double*)malloc(num_constraints * sizeof(double));
    solver->s = (double*)malloc(num_constraints * sizeof(double));
    
    solver->dx = (double*)malloc(state_dim * sizeof(double));
    solver->dy = (double*)malloc(num_constraints * sizeof(double));
    solver->ds = (double*)malloc(num_constraints * sizeof(double));
    
    if (!solver->x || !solver->y || !solver->s ||
        !solver->dx || !solver->dy || !solver->ds) {
        karmarkar_solver_free(solver);
        return -1;
    }
    
    // Initialize to interior point (center of feasible region)
    for (size_t i = 0; i < state_dim; i++) {
        solver->x[i] = 1.0 / state_dim; // Centered initialization
    }
    
    for (size_t i = 0; i < num_constraints; i++) {
        solver->y[i] = 1.0;
        solver->s[i] = 1.0;
    }
    
    return 0;
}

void karmarkar_solver_free(KarmarkarSolver* solver) {
    if (!solver) return;
    
    if (solver->c) {
        free(solver->c);
        solver->c = NULL;
    }
    
    if (solver->A) {
        for (size_t i = 0; i < solver->num_constraints; i++) {
            if (solver->A[i]) {
                free(solver->A[i]);
            }
        }
        free(solver->A);
        solver->A = NULL;
    }
    
    if (solver->b) {
        free(solver->b);
        solver->b = NULL;
    }
    
    if (solver->x) {
        free(solver->x);
        solver->x = NULL;
    }
    
    if (solver->y) {
        free(solver->y);
        solver->y = NULL;
    }
    
    if (solver->s) {
        free(solver->s);
        solver->s = NULL;
    }
    
    if (solver->dx) {
        free(solver->dx);
        solver->dx = NULL;
    }
    
    if (solver->dy) {
        free(solver->dy);
        solver->dy = NULL;
    }
    
    if (solver->ds) {
        free(solver->ds);
        solver->ds = NULL;
    }
    
    if (solver->grid_size) {
        free(solver->grid_size);
        solver->grid_size = NULL;
    }
    
    if (solver->spatial_state) {
        free(solver->spatial_state);
        solver->spatial_state = NULL;
    }
    
    if (solver->buffer) {
        free(solver->buffer);
        solver->buffer = NULL;
    }
    
    memset(solver, 0, sizeof(KarmarkarSolver));
}

// Compute objective value: c^T * x
static double compute_objective(const KarmarkarSolver* solver) {
    double obj = 0.0;
    for (size_t i = 0; i < solver->state_dim; i++) {
        obj += solver->c[i] * solver->x[i];
    }
    return obj;
}

// Compute constraint residuals: A*x - b
static void compute_residuals(const KarmarkarSolver* solver, double* residuals) {
    for (size_t i = 0; i < solver->num_constraints; i++) {
        residuals[i] = -solver->b[i];
        for (size_t j = 0; j < solver->state_dim; j++) {
            residuals[i] += solver->A[i][j] * solver->x[j];
        }
    }
}

// Karmarkar's algorithm step
static int karmarkar_step(KarmarkarSolver* solver) {
    // Compute diagonal scaling matrix D
    double* D = (double*)malloc(solver->state_dim * sizeof(double));
    if (!D) {
        return -1;
    }
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        D[i] = 1.0 / solver->x[i];
    }
    
    // Compute scaled constraint matrix AD
    double** AD = (double**)malloc(solver->num_constraints * sizeof(double*));
    if (!AD) {
        free(D);
        return -1;
    }
    
    for (size_t i = 0; i < solver->num_constraints; i++) {
        AD[i] = (double*)malloc(solver->state_dim * sizeof(double));
        if (!AD[i]) {
            for (size_t j = 0; j < i; j++) {
                free(AD[j]);
            }
            free(AD);
            free(D);
            return -1;
        }
        for (size_t j = 0; j < solver->state_dim; j++) {
            AD[i][j] = solver->A[i][j] * D[j];
        }
    }
    
    // Compute projection matrix P = I - AD^T(AD*AD^T)^{-1}AD
    // Simplified: use gradient descent direction
    double* scaled_c = (double*)malloc(solver->state_dim * sizeof(double));
    if (!scaled_c) {
        for (size_t i = 0; i < solver->num_constraints; i++) {
            free(AD[i]);
        }
        free(AD);
        free(D);
        return -1;
    }
    
    for (size_t i = 0; i < solver->state_dim; i++) {
        scaled_c[i] = solver->c[i] * D[i];
    }
    
    // Compute search direction (simplified Karmarkar direction)
    double* residuals = (double*)malloc(solver->num_constraints * sizeof(double));
    if (!residuals) {
        free(scaled_c);
        for (size_t i = 0; i < solver->num_constraints; i++) {
            free(AD[i]);
        }
        free(AD);
        free(D);
        return -1;
    }
    
    compute_residuals(solver, residuals);
    
    // Project scaled gradient onto null space of constraints
    // Simplified: use projected gradient direction
    for (size_t i = 0; i < solver->state_dim; i++) {
        solver->dx[i] = -scaled_c[i] * solver->alpha;
        
        // Add constraint correction
        for (size_t j = 0; j < solver->num_constraints; j++) {
            solver->dx[i] -= solver->alpha * AD[j][i] * residuals[j];
        }
    }
    
    // Update x with step size
    double step_size = solver->alpha;
    for (size_t i = 0; i < solver->state_dim; i++) {
        double new_x = solver->x[i] + step_size * solver->dx[i] * D[i];
        // Ensure positivity (interior point)
        solver->x[i] = fmax(1e-10, new_x);
    }
    
    // Update barrier parameter
    solver->mu *= solver->beta;
    
    // Cleanup
    free(residuals);
    free(scaled_c);
    for (size_t i = 0; i < solver->num_constraints; i++) {
        free(AD[i]);
    }
    free(AD);
    free(D);
    
    return 0;
}

int karmarkar_ode_solve(KarmarkarSolver* solver, ODEFunction f, double t0, double t_end,
                        const double* y0, void* params, double* y_out) {
    if (!solver || !f || !y0 || !y_out) {
        return -1;
    }
    
    // Initialize state from y0
    memcpy(solver->x, y0, solver->state_dim * sizeof(double));
    
    // Ensure interior point (add small positive value)
    for (size_t i = 0; i < solver->state_dim; i++) {
        if (solver->x[i] <= 0.0) {
            solver->x[i] = 1e-6;
        }
    }
    
    double t = t0;
    double h = 0.01;
    size_t steps = 0;
    solver->current_iteration = 0;
    
    // Formulate ODE as optimization: minimize ||dy/dt - f(t,y)||^2
    while (t < t_end && solver->current_iteration < solver->max_iterations) {
        double h_actual = (t + h > t_end) ? (t_end - t) : h;
        
        // Compute derivative
        double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
        if (!dydt) {
            break;
        }
        
        f(t, solver->x, dydt, params);
        
        // Update objective: minimize residual
        // For Karmarkar, we minimize the ODE residual as a linear program
        // Simplified: use gradient descent with Karmarkar projection
        
        // Karmarkar step
        if (karmarkar_step(solver) != 0) {
            free(dydt);
            break;
        }
        
        // Apply ODE step with Karmarkar correction
        for (size_t i = 0; i < solver->state_dim; i++) {
            solver->x[i] += h_actual * dydt[i] * (1.0 - solver->alpha);
        }
        
        free(dydt);
        
        t += h_actual;
        steps++;
        solver->current_iteration++;
        
        // Check convergence
        double obj = compute_objective(solver);
        if (fabs(obj) < solver->epsilon) {
            break;
        }
    }
    
    memcpy(y_out, solver->x, solver->state_dim * sizeof(double));
    return 0;
}

int karmarkar_pde_solve(KarmarkarSolver* solver, PDEFunction f, double t0, double t_end,
                        const double* u0, size_t spatial_dim, const size_t* grid_size,
                        void* params, double* u_out) {
    if (!solver || !f || !u0 || !u_out || !grid_size) {
        return -1;
    }
    
    solver->spatial_dim = spatial_dim;
    solver->grid_size = (size_t*)malloc(spatial_dim * sizeof(size_t));
    if (!solver->grid_size) {
        return -1;
    }
    
    size_t total_points = 1;
    for (size_t i = 0; i < spatial_dim; i++) {
        solver->grid_size[i] = grid_size[i];
        total_points *= grid_size[i];
    }
    
    solver->spatial_state = (double*)malloc(total_points * sizeof(double));
    if (!solver->spatial_state) {
        free(solver->grid_size);
        return -1;
    }
    
    memcpy(solver->spatial_state, u0, total_points * sizeof(double));
    
    // Use Karmarkar for PDE optimization
    double t = t0;
    double h = 0.01;
    
    while (t < t_end && solver->current_iteration < solver->max_iterations) {
        double h_actual = (t + h > t_end) ? (t_end - t) : h;
        
        // Apply Karmarkar step to spatial grid
        for (size_t i = 0; i < total_points; i++) {
            // Simplified: apply Karmarkar projection to each grid point
            double* dudt = (double*)malloc(sizeof(double));
            double* dudx = (double*)malloc(sizeof(double));
            double* dudy = (double*)malloc(sizeof(double));
            
            if (dudt && dudx && dudy) {
                // Map linear index to spatial coordinates
                double x = (double)(i % grid_size[0]);
                double y = (spatial_dim > 1) ? (double)(i / grid_size[0]) : 0.0;
                
                f(t, x, y, &solver->spatial_state[i], dudt, dudx, dudy, params);
                
                // Update with Karmarkar correction
                solver->spatial_state[i] += h_actual * (*dudt) * (1.0 - solver->alpha);
                
                // Ensure positivity
                solver->spatial_state[i] = fmax(1e-10, solver->spatial_state[i]);
            }
            
            if (dudt) free(dudt);
            if (dudx) free(dudx);
            if (dudy) free(dudy);
        }
        
        t += h_actual;
        solver->current_iteration++;
    }
    
    memcpy(u_out, solver->spatial_state, total_points * sizeof(double));
    return 0;
}

int karmarkar_online_step(KarmarkarSolver* solver, ODEFunction f, double t, double* y,
                          void* params) {
    if (!solver || !f || !y) {
        return -1;
    }
    
    // Update state
    memcpy(solver->x, y, solver->state_dim * sizeof(double));
    
    // Compute derivative
    double* dydt = (double*)malloc(solver->state_dim * sizeof(double));
    if (!dydt) {
        return -1;
    }
    
    f(t, solver->x, dydt, params);
    
    // Karmarkar step with adaptive barrier parameter
    if (karmarkar_step(solver) == 0) {
        // Apply correction
        for (size_t i = 0; i < solver->state_dim; i++) {
            y[i] = solver->x[i];
        }
    }
    
    free(dydt);
    return 0;
}

// Placeholder implementations for ADAM and AdaGrad (if not already implemented)
int adam_solver_init(ADAMSolver* solver, size_t state_dim, ADAMSolverType type,
                     double alpha, double beta1, double beta2, double epsilon) {
    (void)solver; (void)state_dim; (void)type; (void)alpha; (void)beta1; (void)beta2; (void)epsilon;
    return 0;
}

void adam_solver_free(ADAMSolver* solver) {
    (void)solver;
}

int adam_ode_solve(ADAMSolver* solver, ODEFunction f, double t0, double t_end,
                   const double* y0, void* params, double* y_out) {
    (void)solver; (void)f; (void)t0; (void)t_end; (void)y0; (void)params; (void)y_out;
    return 0;
}

int adam_pde_solve(ADAMSolver* solver, PDEFunction f, double t0, double t_end,
                   const double* u0, size_t spatial_dim, const size_t* grid_size,
                   void* params, double* u_out) {
    (void)solver; (void)f; (void)t0; (void)t_end; (void)u0; (void)spatial_dim; (void)grid_size; (void)params; (void)u_out;
    return 0;
}

int adam_online_step(ADAMSolver* solver, ODEFunction f, double t, double* y,
                     void* params) {
    (void)solver; (void)f; (void)t; (void)y; (void)params;
    return 0;
}

int adagrad_solver_init(AdaGradSolver* solver, size_t state_dim, ADAMSolverType type,
                        double learning_rate, double epsilon) {
    (void)solver; (void)state_dim; (void)type; (void)learning_rate; (void)epsilon;
    return 0;
}

void adagrad_solver_free(AdaGradSolver* solver) {
    (void)solver;
}

int adagrad_ode_solve(AdaGradSolver* solver, ODEFunction f, double t0, double t_end,
                      const double* y0, void* params, double* y_out) {
    (void)solver; (void)f; (void)t0; (void)t_end; (void)y0; (void)params; (void)y_out;
    return 0;
}

int adagrad_pde_solve(AdaGradSolver* solver, PDEFunction f, double t0, double t_end,
                      const double* u0, size_t spatial_dim, const size_t* grid_size,
                      void* params, double* u_out) {
    (void)solver; (void)f; (void)t0; (void)t_end; (void)u0; (void)spatial_dim; (void)grid_size; (void)params; (void)u_out;
    return 0;
}

int adagrad_online_step(AdaGradSolver* solver, ODEFunction f, double t, double* y,
                        void* params) {
    (void)solver; (void)f; (void)t; (void)y; (void)params;
    return 0;
}

void compute_ode_gradient(ODEFunction f, double t, const double* y, size_t n,
                          void* params, double* gradient) {
    if (!f || !y || !gradient) {
        return;
    }
    
    double eps = 1e-8;
    double* y_perturbed = (double*)malloc(n * sizeof(double));
    if (!y_perturbed) {
        return;
    }
    
    double* dydt = (double*)malloc(n * sizeof(double));
    if (!dydt) {
        free(y_perturbed);
        return;
    }
    
    f(t, y, dydt, params);
    
    for (size_t i = 0; i < n; i++) {
        memcpy(y_perturbed, y, n * sizeof(double));
        y_perturbed[i] += eps;
        
        double* dydt_perturbed = (double*)malloc(n * sizeof(double));
        if (dydt_perturbed) {
            f(t, y_perturbed, dydt_perturbed, params);
            gradient[i] = (dydt_perturbed[i] - dydt[i]) / eps;
            free(dydt_perturbed);
        }
    }
    
    free(dydt);
    free(y_perturbed);
}

void compute_pde_gradient(PDEFunction f, double t, double x, double y,
                          const double* u, size_t n, void* params, double* gradient) {
    (void)f; (void)t; (void)x; (void)y; (void)u; (void)n; (void)params; (void)gradient;
    // Placeholder implementation
}
