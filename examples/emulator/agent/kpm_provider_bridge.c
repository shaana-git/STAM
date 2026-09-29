#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kpm_provider_bridge.h"

static PyObject *pModule = NULL;
static PyObject *pFuncInit = NULL;
static PyObject *pFuncGetKPM = NULL;
static PyObject *pFuncClose = NULL;

// Convert intensity (0.0-1.0) to string: mild, moderate, severe
static const char* intensity_to_string(double intensity) {
    if (intensity < 0.33) return "mild";
    if (intensity < 0.66) return "moderate";
    return "severe";
}

int init_kpm_provider(const char* scenario_name, const char* attack_type, double intensity) {
    // Initialize Python interpreter
    Py_Initialize();
    
    // Add paths to Python module search path
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");
    PyRun_SimpleString("sys.path.append('./STAM')");
    PyRun_SimpleString("sys.path.append('./STAM/flexric_integration')");
    PyRun_SimpleString("sys.path.append('./STAM/src')");
    
    // Import our provider module
    pModule = PyImport_ImportModule("kpm_data_provider");
    if (!pModule) {
        PyErr_Print();
        fprintf(stderr, "Failed to import kpm_data_provider module\n");
        return -1;
    }
    
    // Get init function (takes 3 args)
    pFuncInit = PyObject_GetAttrString(pModule, "init_provider");
    if (!pFuncInit || !PyCallable_Check(pFuncInit)) {
        PyErr_Print();
        fprintf(stderr, "Failed to get init_provider function\n");
        return -1;
    }
    
    // Get KPM function
    pFuncGetKPM = PyObject_GetAttrString(pModule, "get_kpm_vector");
    if (!pFuncGetKPM || !PyCallable_Check(pFuncGetKPM)) {
        PyErr_Print();
        fprintf(stderr, "Failed to get get_kpm_vector function\n");
        return -1;
    }
    
    // Get close function
    pFuncClose = PyObject_GetAttrString(pModule, "close_provider");
    
    // Map attack type (test_agent.c sends "botnet", "group_ho", "rogue_bs")
    const char* attack_str = "none";
    if (attack_type != NULL) {
        if (strcmp(attack_type, "botnet") == 0) attack_str = "tm1";
        else if (strcmp(attack_type, "group_ho") == 0) attack_str = "tm2";
        else if (strcmp(attack_type, "rogue_bs") == 0) attack_str = "tm3";
        else if (strcmp(attack_type, "tm1") == 0) attack_str = "tm1";
        else if (strcmp(attack_type, "tm2") == 0) attack_str = "tm2";
        else if (strcmp(attack_type, "tm3") == 0) attack_str = "tm3";
    }
    
    const char* intensity_str = intensity_to_string(intensity);
    
    // Call init with scenario, attack_type, intensity
    PyObject *pArgs = PyTuple_New(3);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(scenario_name));
    PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(attack_str));
    PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(intensity_str));
    
    PyObject *pResult = PyObject_CallObject(pFuncInit, pArgs);
    Py_DECREF(pArgs);
    
    if (!pResult) {
        PyErr_Print();
        fprintf(stderr, "Failed to initialize provider\n");
        return -1;
    }
    
    int ret = (int)PyLong_AsLong(pResult);
    Py_DECREF(pResult);
    
    printf("KPM Provider initialized: scenario=%s, attack=%s, intensity=%s\n", 
           scenario_name, attack_str, intensity_str);
    return ret;
}

int get_next_kpm_measurement(float* buffer, int buffer_size) {
    if (!pFuncGetKPM || buffer_size < 26) {
        return -1;
    }
    
    PyObject *pArgs = PyTuple_New(0);
    PyObject *pResult = PyObject_CallObject(pFuncGetKPM, pArgs);
    Py_DECREF(pArgs);
    
    if (!pResult) {
        PyErr_Print();
        return -1;
    }
    
    // Check if result is a list
    if (!PyList_Check(pResult)) {
        Py_DECREF(pResult);
        return -1;
    }
    
    int list_size = PyList_Size(pResult);
    if (list_size < buffer_size) {
        Py_DECREF(pResult);
        return -1;
    }
    
    // Copy values to buffer
    for (int i = 0; i < buffer_size; i++) {
        PyObject *pItem = PyList_GetItem(pResult, i);
        buffer[i] = (float)PyFloat_AsDouble(pItem);
    }
    
    Py_DECREF(pResult);
    return 0;
}

void close_kpm_provider(void) {
    if (pFuncClose && PyCallable_Check(pFuncClose)) {
        PyObject_CallObject(pFuncClose, NULL);
    }
    
    Py_XDECREF(pFuncInit);
    Py_XDECREF(pFuncGetKPM);
    Py_XDECREF(pFuncClose);
    Py_XDECREF(pModule);
    
    Py_Finalize();
    printf("KPM Provider closed\n");
}
