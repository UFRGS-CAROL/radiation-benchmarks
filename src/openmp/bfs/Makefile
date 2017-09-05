# C compiler
CC = g++
CC_FLAGS = -fopenmp -O3 -fpermissive

intel:CC=icpc -mmic -DMIC_NATIVE 
intel:CC_FLAGS=-qopenmp 

SOURCES_CHECK=bfs_check.cpp ../../include/log_helper.c

general: bfs_gen bfs_check

intel: bfs_gen bfs_check

bfs_gen: bfs_gen.cpp 
	$(CC) $(CC_FLAGS) bfs_gen.cpp -o bfs_gen 

bfs_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o bfs_check

clean:
	rm -f bfs_gen bfs_check

