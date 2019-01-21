CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC
EXEC=killtestSignalCPP

INCLUDE= -I./


OBJDIR=./obj/
OBJ= JsonFile.o ClientSocket.o WatchDog.o Log.o Command.o main.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./*.h) Makefile


all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
	
test:
	./$(EXEC) /home/fernando/git_pesquisa/radiation-benchmarks/scripts/json_files/dgemm_check.json