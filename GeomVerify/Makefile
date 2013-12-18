# Makefile for Test

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++0x -I./
LDFLAGS = -L../lib/ -lboost_regex

# Common
CXXFLAGS += -I../Common/include/
LDFLAGS += -lcommon

# KMajority
CXXFLAGS += -I../KMajorityLib/include
LDFLAGS += -lkmajority

# VocabLib
CXXFLAGS += -I../VocabLib/include
LDFLAGS += -lvocab

# OpenCV
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BIN = GeomVerify

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN) $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(BIN) *~
