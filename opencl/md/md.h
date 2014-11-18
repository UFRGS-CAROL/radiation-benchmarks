#ifndef MD_H__
#define MD_H__

#include <cassert>
#include <cfloat>
#include <list>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#include <stdio.h>

struct float4 {
    float x;
    float y;
    float z;
    float w;
};

struct double4 {
    double x;
    double y;
    double z;
    double w;
};

// Problem Constants
static const double  cutsq        = 16.0f; // Square of cutoff distance
static const int     maxNeighbors = 128;  // Max number of nearest neighbors
static const double  domainEdge   = 20.0; // Edge length of the cubic domain
static const double  lj1          = 1.5;  // LJ constants
static const double  lj2          = 2.0;
static const double  EPSILON      = 0.1f; // Relative Error between CPU/GPU

void ocl_alloc_buffers(int nAtom, int maxNeighbors);
void ocl_release_buffers();
void ocl_set_kernel_args(int maxNeighbors, int nAtom);
void ocl_write_position_buffer(int nAtom, double4* position);
void ocl_read_force_buffer(int nAtom, double4 *force);
void ocl_write_neighborList_buffer(int maxNeighbors, int nAtom, int *neighborList);
void ocl_exec_kernel(const long unsigned int global_wsize, const long unsigned int local_wsize);
void deinitOpenCL();
void initOpenCL() ;

#endif
