CXXFLAGS=-L../../../lib -lJTX2Inst -li2c -pthread -lLogHelper

ifdef NANO
CXXFLAGS+=-DNANO

endif

all:
	g++ -I../../../include sample.cpp -o sample_jtx2 $(CXXFLAGS)

clean:
	rm -f sample_jtx2

test:
	./sample_jtx2
