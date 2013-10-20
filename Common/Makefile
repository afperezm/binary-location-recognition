# Makefile for Common

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++11 -fpic -I./
LDFLAGS =

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BINLIB=../lib

LIB = libcommon

all: $(LIB).so

$(LIB).so: $(OBJECTS)
	mkdir -p $(BINLIB)
	$(CXX) -shared $(OBJECTS) -o $(BINLIB)/$@ $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(BINLIB)/$(LIB) *~
