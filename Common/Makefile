# Makefile for Common

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++11 -fpic -I./include/
LDFLAGS =

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BINLIB=../lib

LIB = libcommon

#FILEUTILSVERBOSE = -DFILEUTILSVERBOSE
#DYNMATVERBOSE = -DDYNMATVERBOSE

all: $(LIB).so

$(LIB).so: $(OBJECTS)
	mkdir -p $(BINLIB)
	$(CXX) -shared $(OBJECTS) -o $(BINLIB)/$@ $(LDFLAGS)

.cpp.o:
	$(CXX) $(FILEUTILSVERBOSE) $(DYNMATVERBOSE) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(BINLIB)/$(LIB) *~
