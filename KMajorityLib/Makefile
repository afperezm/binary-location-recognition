# Makefile for K-majority library

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++11 -fpic -I./include/
LDFLAGS = -L../lib/

# OpenCV Extensions
CXXFLAGS += -I../OpenCVExtensions/include
LDFLAGS += -lopencv_extensions

# DBrief
CXXFLAGS += -I../../dbrief_lib
LDFLAGS += -L../../dbrief_lib/lib -ldbrief

# Agast
CXXFLAGS += -I../../agast_lib
LDFLAGS += -L../../agast_lib/lib -lagast

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

#LDFLAGS += -Wl,-rpath=../../agast_lib/lib
#LDFLAGS += -Wl,-rpath=../../dbrief_lib/lib
#LDFLAGS += -Wl,-rpath=../lib/

SOURCES = $(wildcard src/*.cpp)

OBJECTS = $(SOURCES:.cpp=.o)

BINLIB = ../lib

LIBRARY = libkmajority

all: $(LIBRARY).so

$(LIBRARY).so: $(OBJECTS)
	mkdir -p $(BINLIB)
	$(CXX) -shared $(OBJECTS) -o $(BINLIB)/$@ $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(BINLIB)/$(LIBRARY).so *~
