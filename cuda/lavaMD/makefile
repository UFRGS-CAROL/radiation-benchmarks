
ALL: generate chk_gold ecc

generate: lavaMD_generate.cu
	nvcc -arch sm_35 lavaMD_generate.cu -o lava_gen

chk_gold: lavaMD_192_13.cu
	nvcc -arch sm_35 lavaMD_192_13.cu -o lava

ecc: lavaMD_192_13_ecc.cu
	nvcc -arch sm_35 lavaMD_192_13_ecc.cu -o lava_ecc


#clean:

