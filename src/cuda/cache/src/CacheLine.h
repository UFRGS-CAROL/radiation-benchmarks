/*
 * CacheLine.h
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#ifndef CACHELINE_H_
#define CACHELINE_H_

#include <ostream>
#include "utils.h"
#include <vector>
#include "Parameters.h"

#define __CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__

#define CHUNK_SIZE(line_n, t) (line_n / sizeof(t))



/*
 template<uint32 LINE_SIZE, typename memory_type = uint32>
 struct CacheLine {
 //	volatile
 uint32 t[CHUNK_SIZE(LINE_SIZE, memory_type)]; //byte type

 __CUDA_HOST_DEVICE__ CacheLine() {
 }

 __CUDA_HOST_DEVICE__ CacheLine(const CacheLine& T) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 this->t[i] = T.t[i];
 }
 }

 __CUDA_HOST_DEVICE__ CacheLine(const memory_type& T) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T;
 }
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator=(volatile memory_type& T)  {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T;
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator=(memory_type& T) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T;
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator=(CacheLine& T) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T.t[i];
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ volatile CacheLine& operator=(const CacheLine& T) volatile {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T.t[i];
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator=(volatile CacheLine& T) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T.t[i];
 }
 return *this;
 }


 __CUDA_HOST_DEVICE__ CacheLine& operator=(const CacheLine& T) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 t[i] = T.t[i];
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ memory_type operator^(const memory_type& rhs) {
 memory_type ret = rhs;
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 ret ^= t[i];
 }
 return ret;
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator&=(const CacheLine& rhs){
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 this->t[i] &= rhs.t[i];
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator&=(const memory_type& rhs) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 this->t[i] &= rhs;
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ volatile CacheLine& operator&=(const memory_type& rhs) volatile{
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 this->t[i] &= rhs;
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ CacheLine& operator&(const memory_type& rhs) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 this->t[i] &= rhs;
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ volatile CacheLine& operator&(const memory_type& rhs) volatile {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 this->t[i] &= rhs;
 }
 return *this;
 }

 __CUDA_HOST_DEVICE__ bool operator !=(const memory_type& a) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 if (a != t[i])
 return true;
 }
 return false;
 }

 __CUDA_HOST_DEVICE__ bool operator ==(const memory_type& a) {
 #pragma unroll
 for (int i = 0; i < CHUNK_SIZE(LINE_SIZE, memory_type); i++) {
 if (a != t[i])
 return false;
 }
 return true;
 }

 __host__   friend std::ostream& operator<<(std::ostream& stream,
 const CacheLine& t) {
 for (auto s : t.t) {
 stream << " " << s;
 }
 return stream;
 }

 __CUDA_HOST_DEVICE__ memory_type operator [](int idx) const {
 return t[idx];
 }
 };
 */

#endif /* CACHELINE_H_ */
