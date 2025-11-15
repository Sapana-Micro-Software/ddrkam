/*
 * Additional Numerical Methods for ODEs
 * Heun, Midpoint, RK4, and other variants
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef ADDITIONAL_METHODS_H
#define ADDITIONAL_METHODS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * ODE function pointer type
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Heun's Method (Improved Euler)
 * 2nd order method: predictor-corrector
 */
double heun_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params);
size_t heun_solve(ODEFunction f, double t0, double t_end, const double* y0,
                  size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Midpoint Method
 * 2nd order method: uses midpoint evaluation
 */
double midpoint_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params);
size_t midpoint_solve(ODEFunction f, double t0, double t_end, const double* y0,
                      size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Runge-Kutta 4th Order (RK4)
 * Classic 4th order method with 4 stages
 */
double rk4_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params);
size_t rk4_solve(ODEFunction f, double t0, double t_end, const double* y0,
                 size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Ralston's Method
 * 2nd order method with optimal coefficients
 */
double ralston_step(ODEFunction f, double t0, double* y0, size_t n, double h, void* params);
size_t ralston_solve(ODEFunction f, double t0, double t_end, const double* y0,
                     size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Bogacki-Shampine Method (RK23)
 * Adaptive 2nd/3rd order embedded method
 */
double rk23_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                 void* params, double* error_estimate);
size_t rk23_solve(ODEFunction f, double t0, double t_end, const double* y0,
                  size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Dormand-Prince Method (RK45)
 * Adaptive 4th/5th order embedded method
 */
double rk45_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                 void* params, double* error_estimate);
size_t rk45_solve(ODEFunction f, double t0, double t_end, const double* y0,
                  size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Fehlberg Method (RKF45)
 * Adaptive 4th/5th order embedded method
 */
double rkf45_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                  void* params, double* error_estimate);
size_t rkf45_solve(ODEFunction f, double t0, double t_end, const double* y0,
                   size_t n, double h, void* params, double* t_out, double* y_out);

/**
 * Cash-Karp Method (RK45 variant)
 * Adaptive 4th/5th order embedded method
 */
double cash_karp_step(ODEFunction f, double t0, double* y0, size_t n, double h,
                      void* params, double* error_estimate);
size_t cash_karp_solve(ODEFunction f, double t0, double t_end, const double* y0,
                       size_t n, double h, void* params, double* t_out, double* y_out);

#ifdef __cplusplus
}
#endif

#endif /* ADDITIONAL_METHODS_H */
