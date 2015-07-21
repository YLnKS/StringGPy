#ifndef PY_IMPORTS_H
#define PY_IMPORTS_H

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "py_utilities.h"

/* ==== Load some Python functions (typical scipy and numpy functions that are not in math.h) as static objects -- should be called once. ==== */
void load_python_functions(void);

/* ==== Evaluate the modified Bessel function of second kind with fractional order Kv(x) (scipy.special.kv) ==== */
double bessel_kv(double x, double nu);

void initialize_interp(void);

void finalize_interp(void);

#endif