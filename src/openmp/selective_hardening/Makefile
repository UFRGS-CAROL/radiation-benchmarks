# DON'T FORGET TO: source /opt/intel/bin/compilervars.sh intel64
FLAGS = -O2 -fopenmp -lm

all: pi pi_hardened

pi: main.c pi.c
	gcc pi.c main.c -o pi $(FLAGS)

pi_hardened: main.c hardened_pi.c
	gcc hardened_pi.c main.c -o hardened_pi $(FLAGS)

clean:
	rm -f pi hardened_pi *.optrpt *.exec.stackdump 
	
make:
	make clean
	make
