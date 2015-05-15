
ALL: generate chk_gold ecc hard_thread

generate: hotspot_generate.cu
	nvcc -arch sm_35 hotspot_generate.cu -o hotspot_generate

chk_gold: hotspot.cu
	nvcc -arch sm_35 hotspot.cu -o hotspot

ecc: hotspot_ecc.cu
	nvcc -arch sm_35 hotspot_ecc.cu -o hotspot_ecc

hard_thread: hotspot_hard_thread.cu
	nvcc -arch sm_35 hotspot_hard_thread.cu -o hotspot_thread

clean:
	rm hotspot_generate hotspot hotspot_ecc hotspot_thread

