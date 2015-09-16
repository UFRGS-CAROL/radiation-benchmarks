all: cdpAdvancedQuicksort

cdpAdvancedQuicksort: cdpAdvancedQuicksort.cu cdpBitonicSort.cu
	nvcc cdpAdvancedQuicksort.cu cdpBitonicSort.cu -o cdpAdvancedQuicksort -Xcompiler -fopenmp -arch=sm_35 -rdc=true -I.

clean:
	rm -rf cdpAdvancedQuicksort
