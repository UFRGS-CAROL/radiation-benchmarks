CXX=g++ -g
CXX_FLAGS= -O3 -fopenmp -lm

intel:CXX=icpc -mmic -DMIC_NATIVE
intel:CXX_FLAGS=-qopenmp -lm

SOURCES_CHECK=lulesh_check.cc lulesh-util.cc lulesh-init.cc lulesh.h ../../include/log_helper.c
SOURCES_GEN=lulesh_gen.cc lulesh-util.cc lulesh-init.cc lulesh.h 

general: lulesh_gen lulesh_check lulesh_err_inj lulesh_timing

intel: lulesh_gen lulesh_check lulesh_err_inj lulesh_timing

lulesh_gen: $(SOURCES_GEN)
	$(CXX) $(CXX_FLAGS) $(SOURCES_GEN) -o lulesh_gen

lulesh_check: $(SOURCES_CHECK)
	$(CXX) $(CXX_FLAGS) $(SOURCES_CHECK) -DLOGS -o lulesh_check

lulesh_timing: $(SOURCES_CHECK)
	$(CXX) $(CXX_FLAGS) $(SOURCES_CHECK) -DTIMING -o lulesh_timing

lulesh_err_inj: $(SOURCES_CHECK)
	$(CXX) $(CXX_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o lulesh_err_inj

clean:
	rm -f lulesh_gen lulesh_check lulesh_err_inj lulesh_timing
