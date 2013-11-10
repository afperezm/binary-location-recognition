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

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) -o $(CXXFLAGS) -o $(BIN) $(OBJECTS) $(LDFLAGS)

clean:
	rm -rf $(OBJECTS) $(BIN) *~
