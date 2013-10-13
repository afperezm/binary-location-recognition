# Makefile for Common

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++11 -I./include/
LDFLAGS =

# Common
CXXFLAGS = -I../Common
LDFLAGS += -lcommon

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

LIBFOLDER=../lib

LIB = libcommon.a

all: $(LIB)

$(LIB): $(OBJECTS)
	mkdir -p $(LIBFOLDER)
	ar r $(LIB) $(OBJECTS)
	mv $(LIB) $(LIBFOLDER)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(LIBFOLDER)/$(LIB) *~
