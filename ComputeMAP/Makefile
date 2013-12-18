# Makefile ComputeMAP

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++11
LDFLAGS = -L../lib/

# Common
CXXFLAGS += -I../Common/include/
LDFLAGS += -lcommon

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BIN = ComputeMAP

#MAPVERBOSE = -DCMAPVERBOSE

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN) $(LDFLAGS)

.cpp.o:
	$(CXX) $(MAPVERBOSE) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(BIN) *~
