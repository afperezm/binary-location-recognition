# Makefile for K-majority library

CXXFLAGS = -O2 $(DEBUGFLAGS) -fmessage-length=0 -std=c++11 -fpic -I./include/
LDFLAGS = -L../lib/ -lboost_iostreams

# Common
CXXFLAGS += -I../Common/include/
LDFLAGS += -lcommon

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

SOURCES = $(wildcard src/*.cpp)

OBJECTS = $(SOURCES:.cpp=.o)

BINLIB = ../lib

LIBRARY = libkmajority

#SEEDRANDOM = -DSEEDRANDOM
KMAJVERBOSE = -DKMAJVERBOSE

all: $(LIBRARY).so

$(LIBRARY).so: $(OBJECTS)
	mkdir -p $(BINLIB)
	$(CXX) -shared $(OBJECTS) -o $(BINLIB)/$@ $(LDFLAGS)

.cpp.o:
	$(CXX) $(SEEDRANDOM) $(KMAJVERBOSE) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(BINLIB)/$(LIBRARY).so *~
