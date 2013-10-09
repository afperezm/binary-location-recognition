# Makefile for FeatureExtract

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++0x -I../
LDFLAGS =

# DBrief
CXXFLAGS += -I../../dbrief_lib
LDFLAGS += -L../../dbrief_lib/lib -ldbrief

# Agast
CXXFLAGS += -I../../agast_lib
LDFLAGS += -L../../agast_lib/lib -lagast

# AgastFeatureDetector
CXXFLAGS += -I../include

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

LDFLAGS += -Wl,-rpath=../../agast_lib/lib
LDFLAGS += -Wl,-rpath=../../dbrief_lib/lib

SOURCES = FeatureExtract.cpp
OBJS = $(SOURCES:.cpp=.o)
OBJS += ../src/features2d_extensions.o
OBJS += ../src/AgastFeatureDetector.o
OBJS += ../src/DBriefDescriptorExtractor.o

BIN = FeatureExtract

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) -o $(CXXFLAGS) -o $(BIN) $(OBJS) $(LDFLAGS)

clean:
	rm -rf $(OBJS) $(TARGET) *~
