/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
//
//  QUICKSORT.CU
//
//  Implementation of a parallel quicksort in CUDA. It comes in
//  several parts:
//
//  1. A small-set insertion sort. We do this on any set with <=32 elements
//  2. A partitioning kernel, which - given a pivot - separates an input
//     array into elements <=pivot, and >pivot. Two quicksorts will then
//     be launched to resolve each of these.
//  3. A quicksort co-ordinator, which figures out what kernels to launch
//     and when.
//
////////////////////////////////////////////////////////////////////////////////
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "helper_string.h"
#include <sys/time.h>
#include <omp.h>

#include "include/cuda_utils.h"
#include "include/multi_compiler_analysis.h"

#include "cdpQuicksort.h"

#ifdef BUILDPROFILER
#include "include/Profiler.h"
#include "include/NVMLWrapper.h"

#include <memory>

#ifdef FORJETSON
#include "include/JTX2Inst.h"
#define OBJTYPE JTX2Inst
#else
#include "include/NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif // FORJETSON

#endif

int generate;

#ifdef LOGS
#include "log_helper.h"
#endif

#define INPUTSIZE 134217728

typedef struct parameters_s {
	int size;
	int iterations;
	int verbose;
	int debug;
	int generate;
	int fault_injection;
	char *goldName, *inputName;
	unsigned *data, *outdata, *gold;
	unsigned *gpudata, *scratchdata;
	int noinputensurance;
} parameters_t;

void fatal(const char *str) {
	printf("FATAL: %s\n", str);
#ifdef LOGS
	if (generate) end_log_file();
#endif
	exit (EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
// Inline PTX call to return index of highest non-zero bit in a word
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ unsigned int __qsflo(unsigned int word) {
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
	return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  ringbufAlloc
//
//  Allocates from a ringbuffer. Allows for not failing when we run out
//  of stack for tracking the offset counts for each sort subsection.
//
//  We use the atomicMax trick to allow out-of-order retirement. If we
//  hit the size limit on the ringbuffer, then we spin-wait for people
//  to complete.
//
////////////////////////////////////////////////////////////////////////////////
template<typename T>
static __device__ T *ringbufAlloc(qsortRingbuf *ringbuf) {
	// Wait for there to be space in the ring buffer. We'll retry only a fixed
	// number of times and then fail, to avoid an out-of-memory deadlock.
	unsigned int loop = 10000;

	while (((ringbuf->head - ringbuf->tail) >= ringbuf->stacksize) && (loop-- > 0))
		;

	if (loop == 0)
		return NULL;

	// Note that the element includes a little index book-keeping, for freeing later.
	unsigned int index = atomicAdd((unsigned int *) &ringbuf->head, 1);
	T *ret = (T *) (ringbuf->stackbase) + (index & (ringbuf->stacksize - 1));
	ret->index = index;

	return ret;
}

////////////////////////////////////////////////////////////////////////////////
//
//  ringBufFree
//
//  Releases an element from the ring buffer. If every element is released
//  up to and including this one, we can advance the tail to indicate that
//  space is now available.
//
////////////////////////////////////////////////////////////////////////////////
template<typename T>
static __device__ void ringbufFree(qsortRingbuf *ringbuf, T *data) {
	unsigned int index = data->index;       // Non-wrapped index to free
	unsigned int count = atomicAdd((unsigned int *) &(ringbuf->count), 1) + 1;
	unsigned int max = atomicMax((unsigned int *) &(ringbuf->max), index + 1);

	// Update the tail if need be. Note we update "max" to be the new value in ringbuf->max
	if (max < (index + 1))
		max = index + 1;

	if (max == count)
		atomicMax((unsigned int *) &(ringbuf->tail), count);
}

//__ballot
// Replacement
__device__ __forceinline__
unsigned ballout_quicksort(unsigned greater) {
//#if __CUDA_ARCH__ < 600
//  return __ballot(greater);
//#else
	return __ballot_sync(0xFFFFFFFF, greater);
//#endif
}

__device__ __forceinline__
unsigned shift_quicksort(unsigned offset, int lane) {
//#if __CUDA_ARCH__ < 600
//  return __shfl((int) offset, lane);
//#else
	return __shfl_sync(0xFFFFFFFF, offset, lane);
//#endif
}

////////////////////////////////////////////////////////////////////////////////
//
//  qsort_warp
//
//  Simplest possible implementation, does a per-warp quicksort with no inter-warp
//  communication. This has a high atomic issue rate, but the rest should actually
//  be fairly quick because of low work per thread.
//
//  A warp finds its section of the data, then writes all data <pivot to one
//  buffer and all data >pivot to the other. Atomics are used to get a unique
//  section of the buffer.
//
//  Obvious optimisation: do multiple chunks per warp, to increase in-flight loads
//  and cover the instruction overhead.
//
////////////////////////////////////////////////////////////////////////////////
__global__ void qsort_warp(unsigned *indata, unsigned *outdata, unsigned int offset,
		unsigned int len, qsortAtomicData *atomicData, qsortRingbuf *atomicDataStack,
		unsigned int source_is_indata, unsigned int depth) {
	// Find my data offset, based on warp ID
	unsigned int thread_id = threadIdx.x + (blockIdx.x << QSORT_BLOCKSIZE_SHIFT);
	//unsigned int warp_id = threadIdx.x >> 5;   // Used for debug only
	unsigned int lane_id = threadIdx.x & (warpSize - 1);

	// Exit if I'm outside the range of sort to be done
	if (thread_id >= len)
		return;

	//
	// First part of the algorithm. Each warp counts the number of elements that are
	// greater/less than the pivot.
	//
	// When a warp knows its count, it updates an atomic counter.
	//

	// Read in the data and the pivot. Arbitrary pivot selection for now.
	unsigned pivot = indata[offset + len / 2];
	unsigned data = indata[offset + thread_id];

	// Count how many are <= and how many are > pivot.
	// If all are <= pivot then we adjust the comparison
	// because otherwise the sort will move nothing and
	// we'll iterate forever.
	unsigned int greater = (data > pivot);
	unsigned int gt_mask = ballout_quicksort(greater);

	if (gt_mask == 0) {
		greater = (data >= pivot);
		gt_mask = ballout_quicksort(greater); // Must re-ballot for adjusted comparator
	}

	unsigned int lt_mask = ballout_quicksort(!greater);
	unsigned int gt_count = __popc(gt_mask);
	unsigned int lt_count = __popc(lt_mask);

	// Atomically adjust the lt_ and gt_offsets by this amount. Only one thread need do this. Share the result using shfl
	unsigned int lt_offset, gt_offset;

	if (lane_id == 0) {
		if (lt_count > 0)
			lt_offset = atomicAdd((unsigned int *) &atomicData->lt_offset, lt_count);

		if (gt_count > 0)
			gt_offset = len
					- (atomicAdd((unsigned int *) &atomicData->gt_offset, gt_count) + gt_count);
	}

	lt_offset = shift_quicksort(lt_offset, 0); // Everyone pulls the offsets from lane 0
	gt_offset = shift_quicksort(gt_offset, 0);

	__syncthreads();

	// Now compute my own personal offset within this. I need to know how many
	// threads with a lane ID less than mine are going to write to the same buffer
	// as me. We can use popc to implement a single-operation warp scan in this case.
	unsigned lane_mask_lt;
	asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
	unsigned int my_mask = greater ? gt_mask : lt_mask;
	unsigned int my_offset = __popc(my_mask & lane_mask_lt);

	// Move data.
	my_offset += greater ? gt_offset : lt_offset;
	outdata[offset + my_offset] = data;

	// Count up if we're the last warp in. If so, then Kepler will launch the next
	// set of sorts directly from here.
	if (lane_id == 0) {
		// Count "elements written". If I wrote the last one, then trigger the next qsorts
		unsigned int mycount = lt_count + gt_count;

		if (atomicAdd((unsigned int *) &atomicData->sorted_count, mycount) + mycount == len) {
			// We're the last warp to do any sorting. Therefore it's up to us to launch the next stage.
			unsigned int lt_len = atomicData->lt_offset;
			unsigned int gt_len = atomicData->gt_offset;

			cudaStream_t lstream, rstream;
			cudaStreamCreateWithFlags(&lstream, cudaStreamNonBlocking);
			cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);

			// Begin by freeing our atomicData storage. It's better for the ringbuffer algorithm
			// if we free when we're done, rather than re-using (makes for less fragmentation).
			ringbufFree<qsortAtomicData>(atomicDataStack, atomicData);

			// Exceptional case: if "lt_len" is zero, then all values in the batch
			// are equal. We are then done (may need to copy into correct buffer, though)
			if (lt_len == 0) {
				if (source_is_indata)
					cudaMemcpyAsync(indata + offset, outdata + offset, gt_len * sizeof(unsigned),
							cudaMemcpyDeviceToDevice, lstream);

				return;
			}

			// Start with lower half first
			if (lt_len > BITONICSORT_LEN) {
				// If we've exceeded maximum depth, fall through to backup big_bitonicsort
				if (depth >= QSORT_MAXDEPTH) {
					// The final bitonic stage sorts in-place in "outdata". We therefore
					// re-use "indata" as the out-of-range tracking buffer. For (2^n)+1
					// elements we need (2^(n+1)) bytes of oor buffer. The backup qsort
					// buffer is at least this large when sizeof(QTYPE) >= 2.
					big_bitonicsort<<<1, BITONICSORT_LEN, 0, lstream>>>(outdata,
							source_is_indata ? indata : outdata, indata, offset, lt_len);
				} else {
					// Launch another quicksort. We need to allocate more storage for the atomic data.
					if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) == NULL)
						printf("Stack-allocation error. Failing left child launch.\n");
					else {
						atomicData->lt_offset = atomicData->gt_offset = atomicData->sorted_count =
								0;
						unsigned int numblocks = (unsigned int) (lt_len + (QSORT_BLOCKSIZE - 1))
								>> QSORT_BLOCKSIZE_SHIFT;
						qsort_warp<<<numblocks, QSORT_BLOCKSIZE, 0, lstream>>>(outdata, indata,
								offset, lt_len, atomicData, atomicDataStack, !source_is_indata,
								depth + 1);
					}
				}
			} else if (lt_len > 1) {
				// Final stage uses a bitonic sort instead. It's important to
				// make sure the final stage ends up in the correct (original) buffer.
				// We launch the smallest power-of-2 number of threads that we can.
				unsigned int bitonic_len = 1 << (__qsflo(lt_len - 1U) + 1);
				bitonicsort<<<1, bitonic_len, 0, lstream>>>(outdata,
						source_is_indata ? indata : outdata, offset, lt_len);
			}
			// Finally, if we sorted just one single element, we must still make
			// sure that it winds up in the correct place.
			else if (source_is_indata && (lt_len == 1))
				indata[offset] = outdata[offset];

			if (cudaPeekAtLastError() != cudaSuccess)
				printf("Left-side launch fail: %s\n", cudaGetErrorString(cudaGetLastError()));

			// Now the upper half.
			if (gt_len > BITONICSORT_LEN) {
				// If we've exceeded maximum depth, fall through to backup big_bitonicsort
				if (depth >= QSORT_MAXDEPTH)
					big_bitonicsort<<<1, BITONICSORT_LEN, 0, rstream>>>(outdata,
							source_is_indata ? indata : outdata, indata, offset + lt_len, gt_len);
				else {
					// Allocate new atomic storage for this launch
					if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) == NULL)
						printf("Stack allocation error! Failing right-side launch.\n");
					else {
						atomicData->lt_offset = atomicData->gt_offset = atomicData->sorted_count =
								0;
						unsigned int numblocks = (unsigned int) (gt_len + (QSORT_BLOCKSIZE - 1))
								>> QSORT_BLOCKSIZE_SHIFT;
						qsort_warp<<<numblocks, QSORT_BLOCKSIZE, 0, rstream>>>(outdata, indata,
								offset + lt_len, gt_len, atomicData, atomicDataStack,
								!source_is_indata, depth + 1);
					}
				}
			} else if (gt_len > 1) {
				unsigned int bitonic_len = 1 << (__qsflo(gt_len - 1U) + 1);
				bitonicsort<<<1, bitonic_len, 0, rstream>>>(outdata,
						source_is_indata ? indata : outdata, offset + lt_len, gt_len);
			} else if (source_is_indata && (gt_len == 1))
				indata[offset + lt_len] = outdata[offset + lt_len];

			if (cudaPeekAtLastError() != cudaSuccess)
				printf("Right-side launch fail: %s\n", cudaGetErrorString(cudaGetLastError()));
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//  run_quicksort
//
//  Host-side code to run the Kepler version of quicksort. It's pretty
//  simple, because all launch control is handled on the device via CDP.
//
//  All parallel quicksorts require an equal-sized scratch buffer. This
//  must be passed in ahead of time.
//
//  Returns the time elapsed for the sort.
//
////////////////////////////////////////////////////////////////////////////////
double run_quicksort_cdp(parameters_t *params, cudaStream_t stream) {
	double timestamp;
	unsigned int stacksize = QSORT_STACK_ELEMS;

	// This is the stack, for atomic tracking of each sort's status
	qsortAtomicData *gpustack;
	rad::checkFrameworkErrors(cudaMalloc((void **) &gpustack, stacksize * sizeof(qsortAtomicData)));
	rad::checkFrameworkErrors(cudaMemset(gpustack, 0, sizeof(qsortAtomicData))); // Only need set first entry to 0

	// Create the memory ringbuffer used for handling the stack.
	// Initialise everything to where it needs to be.
	qsortRingbuf buf;
	qsortRingbuf *ringbuf;
	rad::checkFrameworkErrors(cudaMalloc((void **) &ringbuf, sizeof(qsortRingbuf)));
	buf.head = 1;           // We start with one allocation
	buf.tail = 0;
	buf.count = 0;
	buf.max = 0;
	buf.stacksize = stacksize;
	buf.stackbase = gpustack;
	rad::checkFrameworkErrors(cudaMemcpy(ringbuf, &buf, sizeof(buf), cudaMemcpyHostToDevice));

	// Timing events...
	timestamp = rad::mysecond();
#ifdef LOGS
	if (!(params->generate)) start_iteration();
#endif

	// Now we trivially launch the qsort kernel
	if (params->size > BITONICSORT_LEN) {
		unsigned int numblocks = (unsigned int) (params->size + (QSORT_BLOCKSIZE - 1))
				>> QSORT_BLOCKSIZE_SHIFT;
		qsort_warp<<<numblocks, QSORT_BLOCKSIZE, 0, stream>>>(params->gpudata, params->scratchdata,
				0U, params->size, gpustack, ringbuf, true, 0);
	} else {
		bitonicsort<<<1, BITONICSORT_LEN>>>(params->gpudata, params->gpudata, 0, params->size);
	}
	rad::checkFrameworkErrors(cudaDeviceSynchronize());

#ifdef LOGS
	if (!(params->generate)) end_iteration();
#endif
	timestamp = rad::mysecond() - timestamp;

	if (cudaGetLastError() != cudaSuccess)
		printf("Launch failure: %s\n", cudaGetErrorString(cudaGetLastError()));
	rad::checkFrameworkErrors(cudaGetLastError());
	// Sanity check that the stack allocator is doing the right thing
	rad::checkFrameworkErrors(cudaMemcpy(&buf, ringbuf, sizeof(*ringbuf), cudaMemcpyDeviceToHost));

	if (params->size > BITONICSORT_LEN && buf.head != buf.tail) {
		printf("Stack allocation error!\nRingbuf:\n");
		printf("\t head = %u\n", buf.head);
		printf("\t tail = %u\n", buf.tail);
		printf("\tcount = %u\n", buf.count);
		printf("\t  max = %u\n", buf.max);
	}

	// Release our stack data once we're done
	rad::checkFrameworkErrors(cudaFree(ringbuf));
	rad::checkFrameworkErrors(cudaFree(gpustack));

	return timestamp;
}

int dataRead(parameters_t *params) {
	FILE *finput = fopen(params->inputName, "rb"), *fgold;
	if (finput) { // READ INPUT
		printf("Reading existing input %s (delete it to generate a new one) ...\n",
				params->inputName);
		double timer = rad::mysecond();
		auto fread_ret = fread(params->data, sizeof(unsigned), params->size, finput);
		if (fread_ret != params->size) {
			printf("Fread different from the expected\n");
			exit (EXIT_FAILURE);
		}
		fclose(finput);
		printf("Done in %.2fs\n", rad::mysecond() - timer);
	} else if (params->generate) { // GENERATE INPUT
		unsigned *ndata = new unsigned[INPUTSIZE];
		printf("Generating input, this will take a long time...");
		fflush (stdout);

		if (!(params->noinputensurance)) {
			printf("Warning: I will alloc %.2fMB of RAM, be ready to crash.",
					(double) UINT_MAX * sizeof(unsigned char) / 1000000);
			printf(
					" If this hangs, it can not ensure the input does not have more than 256 equal entries, which may");
			printf(
					" cause really rare issues under radiation tests. Use -noinputensurance in this case.\n");

			unsigned char *srcHist;
			srcHist = (unsigned char *) malloc(UINT_MAX * sizeof(unsigned char));
			if (!srcHist)
				fatal("Could not alloc RAM for histogram. Use -noinputensurance.");

			register unsigned char max = 0;
			register uint valor;
			for (uint i = 0; i < INPUTSIZE; i++) {
				do {
					valor = rand() % UINT_MAX;
				} while (srcHist[valor] == UCHAR_MAX);
				srcHist[valor]++;
				if (srcHist[valor] > max)
					max = srcHist[valor];
				ndata[i] = valor;
			}
			free(srcHist);
			printf("Maximum repeats of one single key: %d\n", max);
		} else {
			for (unsigned int i = 0; i < INPUTSIZE; i++) {
				// Build data 8 bits at a time
				ndata[i] = 0;
				char *ptr = (char *) &(ndata[i]);

				for (unsigned j = 0; j < sizeof(unsigned); j++) {
					*ptr++ = (char) (rand() & 255);
				}
			}
		}
		if (!(finput = fopen(params->inputName, "wb"))) {
			printf("Warning! Couldn't write the input to file, proceeding anyway...\n");
		} else {
			fwrite(ndata, INPUTSIZE * sizeof(unsigned), 1, finput);
			fclose(finput);
		}
		memcpy(params->data, ndata, params->size * sizeof(unsigned));
		printf("Done.\n");
	} else {
		fatal("Input file not opened. Use -generate.\n");
	}

	if (!(params->generate)) {
		if (!(fgold = fopen(params->goldName, "rb")))
			fatal("Gold file not opened. Use -generate.\n");
		auto fread_ret = fread(params->gold, sizeof(unsigned), params->size, fgold);
		if (fread_ret != params->size) {
			printf("Fread different from the expected\n");
			exit (EXIT_FAILURE);
		}
		fclose(fgold);
	}

	if (params->fault_injection) {
		params->data[6] = rand();
		printf(">>>>>>> Error injection: data[6]=%i\n", params->data[6]);
	}
	return 1;
}

void goldWrite(parameters_t *params) {
	FILE *fgold;
	if (!(fgold = fopen(params->goldName, "wb"))) {
		printf("Gold file could not be open in wb mode.\n");
	} else {
		fwrite(params->gold, params->size * sizeof(unsigned), 1, fgold);
		fclose(fgold);
	}
}

void outputWrite(parameters_t *params, char *fname) {
	FILE *fout;
	if (!(fout = fopen(fname, "wb"))) {
		printf("Output file could not be open in wb mode.\n");
	} else {
		fwrite(params->outdata, params->size * sizeof(unsigned), 1, fout);
		fclose(fout);
	}
}

int checkKeys(parameters_t *params) { // Magicas que a semana anterior ao teste proporcionam
	register unsigned char *srcHist;
	register unsigned char *resHist;

	uint numValues = UINT_MAX;

	int flag = 1;
	int errors = 0;

	register uint index, range;
	long unsigned int control;
	range = ((numValues * sizeof(unsigned char) > 2048000000) ? 2048000000 : numValues); // Avoid more than 1GB of RAM alloc

	srcHist = (unsigned char *) malloc(range * sizeof(unsigned char));
	resHist = (unsigned char *) malloc(range * sizeof(unsigned char));

	if (!srcHist || !resHist)
		fatal("Could not alloc src or res");

	for (index = 0, control = 0; control < numValues; index += range, control += range) {
		printf("index = %u range = %u alloc=%.2fMB\n", index, range,
				2 * (double) range * sizeof(unsigned char) / 1000000);

		//Build histograms for keys arrays
		memset(srcHist, 0, range * sizeof(unsigned char));
		memset(resHist, 0, range * sizeof(unsigned char));

		register uint indexPLUSrange = index + range;
		register uint *srcKey = params->data;
		register uint *resKey = params->outdata;
		register uint i;
#pragma omp parallel for
		for (i = 0; i < params->size; i++) {
			//if (index!=0) printf("srcKey[%d]=%d resKey[%d]=%d index=%d indexPLUSrange=%d\n", i, srcKey[i], i, resKey[i], index, indexPLUSrange); fflush(stdout);
			if ((srcKey[i] >= index) && (srcKey[i] < indexPLUSrange) && (srcKey[i] < numValues)) {
#pragma omp atomic
				srcHist[srcKey[i] - index]++;
			}
			if ((resKey[i] >= index) && (resKey[i] < indexPLUSrange) && (resKey[i] < numValues)) {
#pragma omp atomic
				resHist[resKey[i] - index]++;
			}
		}
#pragma omp parallel for
		for (i = 0; i < range; i++)
			if (srcHist[i] != resHist[i])
#pragma omp critical
					{
				char error_detail[150];
				snprintf(error_detail, 150,
						"The histogram from element %d differs. srcHist=%d dstHist=%d\n", i + index,
						srcHist[i], resHist[i]);
#ifdef LOGS
				if (!(params->generate)) log_error_detail(error_detail);
#endif
				if (errors <= 10)
					printf("ERROR : %s\n", error_detail);
				errors++;
				flag = 0;
			}

	}
	free(resHist);
	free(srcHist);

	//Finally check the ordering
	register uint *resKey = params->outdata;
	register uint i;
#pragma omp parallel for
	for (i = 0; i < params->size - 1; i++)
		if (resKey[i] > resKey[i + 1])
#pragma omp critical
				{
			char error_detail[150];
			snprintf(error_detail, 150, "Elements not ordered. index=%d %d>%d", i, resKey[i],
					resKey[i + 1]);
#ifdef LOGS
			if (!(params->generate)) log_error_detail(error_detail);
#endif
			if (errors <= 10)
				printf("ERROR : %s\n", error_detail);
			errors++;
			flag = 0;
		}

	if (flag)
		printf("OK\n");
	if (!flag)
		printf("Errors found.\n");

	return errors;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int run_qsort(parameters_t *params) {
	srand (time(NULL));

int	size = params->size;

	// Create CPU data.
	params->data = new unsigned[size];
	params->outdata = new unsigned[size];
	params->gold = new unsigned[size];

	dataRead(params);// Read data from files

	int loop1;// Test loop iterator
	for (loop1 = 0; loop1 < params->iterations; loop1++) {
		double itertimestamp = rad::mysecond();
		if (params->verbose)
		printf("================== [Iteration #%i began]\n", loop1);

		// Create and set up our test
		rad::checkFrameworkErrors(
				cudaMalloc((void **) &(params->gpudata),
						size * sizeof(unsigned)));
		rad::checkFrameworkErrors(
				cudaMalloc((void **) &(params->scratchdata),
						size * sizeof(unsigned)));

		rad::checkFrameworkErrors(
				cudaMemcpy(params->gpudata, params->data,
						size * sizeof(unsigned), cudaMemcpyHostToDevice));

		// So we're now populated and ready to go! We size our launch as
		// blocks of up to BLOCKSIZE threads, and appropriate grid size.
		// One thread is launched per element.
		double ktime = run_quicksort_cdp(params, NULL);

		if (params->verbose)
		printf("GPU Kernel time: %.4fs\n", ktime);

		// Copy back the data and verify correct sort
		rad::checkFrameworkErrors(
				cudaMemcpy(params->outdata, params->gpudata,
						params->size * sizeof(unsigned),
						cudaMemcpyDeviceToHost));

		double timer = rad::mysecond();
		int errors = 0;

		if (params->generate) {        // Write gold to file
			printf("Verify gold consistence...\n");
			errors = checkKeys(params);
			printf("Writing gold to file %s...\n", params->goldName);
			memcpy(params->gold, params->outdata, size * sizeof(unsigned));
			goldWrite(params);
			printf("Done.\n");
		} else {
			if (memcmp(params->gold, params->outdata,
							size * sizeof(unsigned))) {
				printf(
						"Warning! Gold file mismatch detected, proceeding to error analysis...\n");

				errors = checkKeys(params);
			} else {
				errors = 0;
			}
#ifdef LOGS
			if (!(params->generate)) log_error_count(errors);
#endif
		}

		if (params->verbose)
		printf("Gold check/generate time: %.4fs\n", rad::mysecond() - timer);

		// Release everything and we're done
		rad::checkFrameworkErrors(cudaFree(params->scratchdata));
		rad::checkFrameworkErrors(cudaFree(params->gpudata));

		// Display the time between event recordings
		if (params->verbose)
		printf("Perf: %.3fM elems/sec\n", 1.0e-6f * size / ktime);
		if (params->verbose) {
			printf("Iteration %d ended (Errors: %d). Elapsed time: %.4fs\n",
					loop1, errors, rad::mysecond() - itertimestamp);
		} else {
			printf(".");
		}
		fflush(stdout);
	}

	delete (params->data);
	return 0;
}

static void usage(int argc, char *argv[]) {
	printf(
			"Syntax: %s -size=N [-generate] [-verbose] [-debug] [-input=<path>] [-gold=<path>] [-iterations=N] [-noinputensurance]\n",
			argv[0]);
	exit (EXIT_FAILURE);
}

void getParams(int argc, char *argv[], parameters_t *params) {
	params->size = 5000;
	params->iterations = 100000000;
	params->verbose = 0;
	params->generate = 0;
	params->fault_injection = 0;
	params->noinputensurance = 0;
	generate = 0;

	if (checkCmdLineFlag(argc, (const char **) argv, "help")
			|| checkCmdLineFlag(argc, (const char **) argv, "h")) {
		usage(argc, argv);
		printf("&&&& cdpAdvancedQuicksort WAIVED\n");
		exit (EXIT_WAIVED);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
		params->size = getCmdLineArgumentInt(argc, (const char **) argv, "size");
		if (params->size > INPUTSIZE) {
			fatal(
					"Maximum size reached, please increase the input size on the code source and recompile.");
		}
	} else {
		printf("Missing -size parameter.\n");
		usage(argc, argv);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		params->verbose = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		params->generate = 1;
		generate = 1;
		params->iterations = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "noinputensurance")) {
		params->noinputensurance = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		params->iterations = getCmdLineArgumentInt(argc, (const char **) argv, "iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
		params->fault_injection = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold", &(params->goldName));
	} else {
		params->goldName = new char[100];
		snprintf(params->goldName, 100, "quickSortGold%i", (signed int) params->size);
		printf("Using default gold filename: %s\n", params->goldName);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input", &(params->inputName));
	} else {
		params->inputName = new char[100];
		snprintf(params->inputName, 100, "quickSortInput%i", (signed int) INPUTSIZE);
		printf("Using default input filename: %s\n", params->inputName);
	}
}

// Host side entry
int main(int argc, char *argv[]) {
	parameters_t *params;

	params = (parameters_t*) malloc(sizeof(parameters_t));

	getParams(argc, argv, params);

	// Get device properties
	int cuda_device = findCudaDevice(argc, (const char **) argv);
	cudaDeviceProp properties;
	rad::checkFrameworkErrors(cudaGetDeviceProperties(&properties, cuda_device));
	int cdpCapable = (properties.major == 3 && properties.minor >= 5) || properties.major >= 4;

	printf("GPU device %s has compute capabilities (SM %d.%d)\n", properties.name, properties.major,
			properties.minor);

	if (!cdpCapable) {
		printf(
				"cdpAdvancedQuicksort requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...\n");
		exit (EXIT_WAIVED);
	}

	std::string test_info = "size:" + std::to_string(params->size);
	test_info += rad::get_multi_compiler_header();
#ifdef LOGS
	if (!(params->generate)) start_log_file(const_cast<char*>("cudaQuickSortCDP"),
			const_cast<char*>(test_info.c_str()));
#ifdef BUILDPROFILER
	auto str = std::string(get_log_file_name());
	if(params->generate) {
		str = "/tmp/generate.log";
	}
	auto profiler_thread = std::make_shared<rad::OBJTYPE>(0, str);

	//START PROFILER THREAD
	profiler_thread->start_profile();
#endif

#endif

	printf("Running qsort on %d elements, on %s\n", params->size, properties.name);

	run_qsort(params);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	rad::checkFrameworkErrors(cudaDeviceReset());
#ifdef LOGS
#ifdef BUILDPROFILER
	profiler_thread->end_profile();
#endif

	if (!(params->generate)) end_log_file();
#endif
	exit (EXIT_SUCCESS);
}
