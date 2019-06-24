
all:
	g++ -I../../../include sample.cpp -o sample_jtx2 -L../../../lib -lJTX2Inst -li2c -pthread 

clean:
	rm -f sample_jtx2

test:
	./sample_jtx2
