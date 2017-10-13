FLAGS := -Wno-write-strings

all: hotspotex hotspotver

hotspotex: hotspotex.cpp
	g++ $(FLAGS) -D__$(SIZE)__ -o $@ $<

hotspotver: hotspotver.cpp
	g++ $(FLAGS) -D__$(SIZE)__ -o $@ $<

clean:
	rm -rf hotspotex hotspotver

run: clean hotspotex hotspotver
	./hotspotex
	./hotspotver
