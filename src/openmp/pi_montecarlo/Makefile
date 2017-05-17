# DON'T FORGET TO: source /opt/intel/bin/compilervars.sh intel64
all: pi pi_hardened

pi: main.c pi.c
	gcc pi.c main.c -o pi -O2 -fopenmp -lm

pi_hardened: main.c pi_hardened.c
	gcc pi_hardened.c main.c -o pi_hardened -O0 -fopenmp -lm

clean:
	rm -f pi pi_hardened *.optrpt *.exec.stackdump 
	
make:
	make clean
	make
