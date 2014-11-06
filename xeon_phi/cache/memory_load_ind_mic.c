/*
 * Copyright (C) 2014  Marco Antonio Zanata Alves (mazalves at inf.ufrgs.br)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

// Xeon Phi total cores = 57. 1 core probably runs de OS.
#define MIC_NUM_THREADS 1
// #define ARRAY_SIZE 56000
// #define MAX 32000
#define REFWORD 1 //  0x55555555


///=============================================================================
int main (int argc, char *argv[]) {

    uint32_t size=0;
    uint32_t repetitions=0;

    if(argc != 3) {
        printf("Please provide the number of repetitions and array size.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = atoi(argv[1]);
    size = atoi(argv[2]);

    if (size % 32 != 0) {
        printf("The array size needs to be divisible by 32 (due to unrolling).\n");
        exit(EXIT_FAILURE);
    }
    printf("Element size %"PRIu32"B\n", (uint32_t)sizeof(uint32_t));
    printf("Repetitions:%"PRIu32" Size:%"PRIu32"KB\n", repetitions, size / 1024);
    printf("Elements to be accessed: %"PRIu32"K\n", (uint32_t)(size / sizeof(uint32_t)) / 1024);

    omp_set_num_threads(MIC_NUM_THREADS);
    printf("Threads:%"PRIu32"\n", MIC_NUM_THREADS);

    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t jump = 0;
    uint32_t count = 0;
    uint32_t slice = (size / sizeof(uint32_t)) / MIC_NUM_THREADS ;

    uint32_t *ptr_vector;
    ptr_vector = (uint32_t *)valloc(size);

    for (i = 0; i < (size / sizeof(uint32_t)); i++) {
        ptr_vector[i] = REFWORD;
    }

    #pragma offload target(mic) in(ptr_vector:length(size / sizeof(uint32_t))) reduction(+:count)
    {
        #pragma omp parallel for private(i, j, jump)
        for(j = 0; j < MIC_NUM_THREADS; j++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

	    uint32_t th_id = omp_get_thread_num();
            for (i = 0; i < repetitions; i++) {
                for (jump = slice * th_id; jump < slice * (th_id + 1) ;){

		    // DEBUG: injecting one error
		    //if(j == 0 && i == 0 && jump == 0)
		    //    ptr_vector[jump] = 0;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;

		    if ((ptr_vector[jump] ^ REFWORD)) {
			count++;
			printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, j, ptr_vector[jump] ^ REFWORD);
			ptr_vector[jump] = REFWORD;
		    }
		    jump++;


                }
            }
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("%"PRIu32"\n", count);
    exit(EXIT_SUCCESS);
}
