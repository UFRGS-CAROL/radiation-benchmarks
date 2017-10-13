/*
 * maxpool_layer_hardened_kernels.cu
 *
 *  Created on: 17/05/2017
 *      Author: fernando
 */

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}

#define FACTOR 10.0

unsigned long long *error_detected = NULL;

float LOOK_UP_TABLE[] = { //for hardened maxpool
        34.8208, // layer  0
                34.8208, // layer  1
                25.9040, // layer  2
                25.9040, // layer  3
                20.1720, // layer  4
                26.2095, // layer  5
                24.7247, // layer  6
                24.7247, // layer  7
                22.0913, // layer  8
                31.6815, // layer  9
                31.0876, // layer  10
                31.0876, // layer  11
                28.7064, // layer  12
                53.9315, // layer  13
                28.5885, // layer  14
                30.7862, // layer  15
                20.9733, // layer  16
                20.9733, // layer  17
                19.6744, // layer  18
                44.5123, // layer  19
                19.7984, // layer  20
                40.2696, // layer  21
                102.170, // layer  22
                20.5588, // layer  23
                22.0682, // layer  24
                22.0682, // layer  25
                19.1314, // layer  26
                19.1314, // layer  27
                19.1314, // layer  28
                17.6953, // layer  29
                43.3081, // layer  30
                43.3081 //layer 31
        };

int maxpool_iterator = 0;

__global__ void forward_maxpool_layer_kernel_hardened(int n, int in_h, int in_w,
        int in_c, int stride, int size, int pad, float *input, float *output,
        int *indexes, float max_value_allowed, unsigned long long *error_detected, int maxp) {
    int h = (in_h + 2 * pad) / stride;
    int w = (in_w + 2 * pad) / stride;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;

    int out_index = j + w * (i + h * (k + c * b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;

//temp matrix
    float old_max = max;
    int old_max_i = max_i;

    for (l = 0; l < size; ++l) {
        for (m = 0; m < size; ++m) {
            int cur_h = h_offset + i * stride + l;
            int cur_w = w_offset + j * stride + m;
            int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            int valid = (cur_h >= 0 && cur_h < in_h && cur_w >= 0
                    && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;

            max_i = (val > max) ? index : max_i;
            max = (val > max) ? val : max;
            //hardened trick
            if (max > max_value_allowed) {
                max = old_max;
                max_i = old_max_i;

                //count how many errors
                atomicAdd(&error_detected[maxp], 1);
            } else {
                old_max = max;
                old_max_i = max_i;
            }

        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

void forward_maxpool_layer_gpu_hardened(maxpool_layer layer, network net) {
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h * w * c * layer.batch;

//for the LOOKUP
    maxpool_iterator = (maxpool_iterator + 1) % MAXPOOL_N;
    int maxp = 1;
    if (maxpool_iterator == 1) {
        maxp = 3;
    } else if (maxpool_iterator == 2) {
        maxp = 7;
    } else if (maxpool_iterator == 3) {
        maxp = 11;
    } else if (maxpool_iterator == 4) {
        maxp = 17;
    }

    if(error_detected == NULL){
        cudaMalloc(&error_detected, sizeof(unsigned long long) * MAXPOOL_N);
    }

    forward_maxpool_layer_kernel_hardened<<<cuda_gridsize(n), BLOCK>>>(n,
            layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad,
            net.input_gpu, layer.output_gpu, layer.indexes_gpu,
            LOOK_UP_TABLE[maxp] * FACTOR, error_detected, maxpool_iterator);
    check_error(cudaPeekAtLastError());
}

__global__ void memset_error(unsigned long long *error_detected){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    error_detected[i] = 0;
}

/**
 * host_error_detected must be allocated
 */
void get_and_reset_error_detected_values(error_return *host_error) {
    //copy from error_detected var
    cudaMemcpy(host_error->error_detected, error_detected,
            sizeof(unsigned long long) * host_error->err_detected_size, cudaMemcpyDeviceToHost);

//    memset_error<<<1, MAXPOOL_N>>>(error_detected);
    cudaMemset(error_detected, 0, sizeof(unsigned long long) * host_error->err_detected_size);

    check_error(cudaPeekAtLastError());
}

void free_err_detected(){
    if(error_detected)
        cudaFree(error_detected);
}
