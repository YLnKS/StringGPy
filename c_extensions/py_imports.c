#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cStringGPy_ARRAY_API

#include "py_imports.h"

static PyObject *py_kv = NULL;
static PyObject *py_kvp = NULL;

void load_python_functions()
{
	if(!Py_IsInitialized())
    	Py_Initialize();

    PyObject *spName, *spModule;	
    spName = PyString_FromString("scipy.special");
    /* Error checking of spName left out */
    spModule = PyImport_Import(spName);
    Py_DECREF(spName);

    if (spModule != NULL) {
        py_kv = PyObject_GetAttrString(spModule, "kv");
        py_kvp = PyObject_GetAttrString(spModule, "kvp");
        /* pFunc is a new reference */
        Py_DECREF(spModule);
    }
    else {
    	PyErr_SetString(PyExc_ValueError, "Failed to load scipy.special.");
    }
}


void initialize_interp(){
	load_python_functions();
}


void finalize_interp(){
	Py_DECREF(py_kv);
	Py_DECREF(py_kvp);

	py_kv=NULL;
	py_kvp=NULL;
}

/* ==== Evaluate the modified Bessel function of second kind with fractional order Kv(x) (scipy.special.kv) ==== */
double bessel_kv(double nu, double x){
	double res;
	PyObject *pArgs, *pValue, *p_nu, *p_x;
	
	if(py_kv == NULL){
		load_python_functions();
	}
	
	/* Doubles need to be converted to use the Python function */
	p_nu = PyFloat_FromDouble(nu);
	p_x = PyFloat_FromDouble(x);
	
	/* Construct Python arguments */
	pArgs = PyTuple_New(2);
	PyTuple_SetItem(pArgs, 0, p_nu);
	PyTuple_SetItem(pArgs, 1, p_x);

	/* Call the modified Bessel function from Python */
	pValue = PyObject_CallObject(py_kv, pArgs);
	
	/* Convert the python result as double */
	res = PyFloat_AsDouble(pValue);	
	
	/* Release references */
	Py_DECREF(p_nu);
	Py_DECREF(p_x);
    Py_DECREF(pArgs);
	Py_DECREF(pValue);
	
	return res;	
}