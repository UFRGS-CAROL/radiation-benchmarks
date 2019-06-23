
all:
	g++ -I../../../include ../src/*.cpp ../../JTX2Inst.cpp sample.cpp -li2c -pthread -o sample_jtx2

clean:
	rm -f sample_jtx2

test:
	./sample_jtx2
