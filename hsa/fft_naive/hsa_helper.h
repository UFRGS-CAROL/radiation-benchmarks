#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "elf_utils.h"

int init_hsa(char *kernel_name, char *brig_name);

void set_kernel_dim(int group_size_x, int grid_size_x);

void set_args(void *arg1, int size_arg1, void *arg2, int size_arg2, void *arg3, int size_arg3);

void run_kernel();

void clean_hsa_resources();

