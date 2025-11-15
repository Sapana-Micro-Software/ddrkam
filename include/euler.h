/*
 * Euler's Method for Ordinary Differential Equations
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef EULER_H
#define EULER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * Function pointer type for the differential equation system
 * @param t: Current time
 * @param y: Current state vector
 * @param dydt: Output derivative vector
 * @param params: User-defined parameters
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Euler's Method (1st order)
 * y_{n+1} = y_n + h * f(t_n, y_n)
 * 
 * @param f: Function defining the ODE system
 * @param t0: Initial time
 * @param y0: Initial state vector (input/output)
 * @param n: Dimension of the system
 * @param h: Step size
 * @param params: User-defined parameters for the ODE function
 * @return: New time value (t0 + h)
 */
double euler_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params);

/**
 * Solve ODE system using Euler's method over a time interval
 * 
 * @param f: Function defining the ODE system
 * @param t0: Initial time
 * @param t_end: Final time
 * @param y0: Initial state vector
 * @param n: Dimension of the system
 * @param h: Step size
 * @param params: User-defined parameters
 * @param t_out: Output time array (allocated by caller)
 * @param y_out: Output state array (n x num_steps, allocated by caller)
 * @return: Number of steps taken
 */
size_t euler_solve(ODEFunction f, double t0, double t_end, const double* y0, 
                   size_t n, double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* EULER_H */
