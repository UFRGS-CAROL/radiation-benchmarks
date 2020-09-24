//
// Created by fernando on 9/24/20.
//

#ifndef CACHE_DEVICE_FUNCTIONS_H
#define CACHE_DEVICE_FUNCTIONS_H

__device__ __forceinline__ static
void sleep_cuda(const int64 &clock_count) {
    const int64 start = clock64();
    while ((clock64() - start) < clock_count);
}


__device__ __forceinline__ static
void move_cache_line(uint64 *dst, uint64 *src) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    dst[4] = src[4];
    dst[5] = src[5];
    dst[6] = src[6];
    dst[7] = src[7];
    dst[8] = src[8];
    dst[9] = src[9];
    dst[10] = src[10];
    dst[11] = src[11];
    dst[12] = src[12];
    dst[13] = src[13];
    dst[14] = src[14];
    dst[15] = src[15];
}


#endif //CACHE_DEVICE_FUNCTIONS_H
