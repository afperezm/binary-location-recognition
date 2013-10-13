# Makefile for VocabLib

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++0x
LDFLAGS = -L../lib/

# OpenCV Extensions
CXXFLAGS += -I../OpenCVExtensions/include
LDFLAGS += -lopencv_extensions

# Common
CXXFLAGS += -I../Common/
LDFLAGS += -lcommon

# DBrief
CXXFLAGS += -I../../dbrief_lib
LDFLAGS += -L../../dbrief_lib/lib -ldbrief

# Agast
CXXFLAGS += -I../../agast_lib
LDFLAGS += -L../../agast_lib/lib -lagast

# KMajority
CXXFLAGS += -I../KMajorityLib/include
LDFLAGS += -lkmajority

# VocabLib
CXXFLAGS += -I../VocabLib/include
LDFLAGS += -lvocab

# DBow
CXXFLAGS += -I../../DBoW2/DUtils -I../../DBoW2/DUtilsCV -I../../DBoW2/DVision -I../../DBoW2/DBoW2
LDFLAGS += -L../../DBoW2/lib -lDBoW2 -lDUtils -lDUtilsCV -lDVision -lstdc++

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

LDFLAGS += -Wl,-rpath=../../agast_lib/lib
LDFLAGS += -Wl,-rpath=../../dbrief_lib/lib
LDFLAGS += -Wl,-rpath=../../DBoW2/lib
LDFLAGS += -Wl,-rpath=../lib/

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BIN = VocabLearn

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) -o $(CXXFLAGS) -o $(BIN) $(OBJECTS) $(LDFLAGS)

clean:
	rm -rf $(OBJECTS) $(BIN) *~
