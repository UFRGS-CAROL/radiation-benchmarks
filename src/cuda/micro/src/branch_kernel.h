#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

#include <cstdint>


__global__ void int_branch_kernel(int32_t* dst_1, int32_t* dst_2, int32_t* dst_3, uint32_t op);

#endif /* BRANCH_KERNEL_H_ */
