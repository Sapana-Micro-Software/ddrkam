/*
 * Adams-Bashforth and Adams-Moulton Methods
 * Copyright (C) 2025, Shyamal Suhana Chandra
 */

#ifndef ADAMS_H
#define ADAMS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * Function pointer type for the differential equation system
 */
typedef void (*ODEFunction)(double t, const double* y, double* dydt, void* params);

/**
 * Adams-Bashforth 3rd order method (predictor)
 * 
 * @param f: Function defining the ODE system
 * @param t: Time array (at least 3 elements)
 * @param y: State array (n x 3, column-major)
 * @param n: Dimension of the system
 * @param h: Step size
 * @param params: User-defined parameters
 * @param y_pred: Output predicted state
 */
void adams_bashforth3(ODEFunction f, const double* t, const double* y, 
                      size_t n, double h, void* params, double* y_pred);

/**
 * Adams-Moulton 3rd order method (corrector)
 * 
 * @param f: Function defining the ODE system
 * @param t: Time array (at least 3 elements)
 * @param y: State array (n x 3, column-major)
 * @param n: Dimension of the system
 * @param h: Step size
 * @param params: User-defined parameters
 * @param y_pred: Predicted state (input)
 * @param y_corr: Output corrected state
 */
void adams_moulton3(ODEFunction f, const double* t, const double* y, 
                    size_t n, double h, void* params, 
                    const double* y_pred, double* y_corr);

#ifdef __cplusplus
}
#endif

#endif /* ADAMS_H */
