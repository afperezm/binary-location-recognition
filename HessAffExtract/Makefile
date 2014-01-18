# Makefile for Test

CXX = g++

CXXFLAGS = -O2 -Wall -fmessage-length=0 -std=c++0x
LDFLAGS = -L../lib/

# Common
CXXFLAGS += -I../Common/include/
LDFLAGS += -lcommon

# OpenCV
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BIN = HessAffExtract

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN) $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(BIN) *~
