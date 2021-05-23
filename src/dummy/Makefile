CXX = g++
EXEC = dummy

SRC = dummy_application_for_test.cpp

INCLUDE = -I../cuda/common -I../include
LDFLAGS = -lLogHelper  -L../include 

CXXFLAGS = -std=c++11 -O3 $(INCLUDE) -DLOGS


all: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)
 

clean:
	rm $(EXEC)
