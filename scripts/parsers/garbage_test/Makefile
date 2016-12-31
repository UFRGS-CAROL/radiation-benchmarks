all: clean test



test: 
	gcc -c log_processing.c -o log_processing.o
	gcc log_processing.o TestGoldContentC.c -o test 

clean:
	rm -rf test
