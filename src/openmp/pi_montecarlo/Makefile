# DON'T FORGET TO: source /opt/intel/bin/compilervars.sh intel64
FLAGS = -O2 -fopenmp -lm

all: pi pi_hardened

pi: main.c pi.c
	gcc pi.c main.c -o pi $(FLAGS)

pi_hardened: main.c pi_hardened.c
	gcc pi_hardened.c main.c -o pi_hardened $(FLAGS)

clean:
	rm -f pi pi_hardened *.optrpt *.exec.stackdump 
	
make:
	make clean
	make
