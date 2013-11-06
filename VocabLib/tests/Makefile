# Makefile for VocabLib Tests

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++0x
LDFLAGS = -L../../lib/

# KMajority
CXXFLAGS += -I../../KMajorityLib/include
LDFLAGS += -lkmajority

# VocabLib
CXXFLAGS += -I../include
LDFLAGS += -lvocab

# Common
CXXFLAGS += -I../../Common/
LDFLAGS += -lcommon

# OpenCV
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

# GoogleTest
CXXFLAGS += -Wextra -pthread
LDFLAGS += -L/home/andresf/workspace-cpp/gtest-1.7.0/make
LDFLAGS += -lgtest_main -lpthread

SOURCES = VocabTree_test.cpp
OBJECTS = $(SOURCES:.cpp=.o)

BIN =  VocabTree_test

#VTREEVERBOSE = -DVTREEVERBOSE
#DYNMATVERBOSE = -DDYNMATVERBOSE

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) $(VTREEVERBOSE) $(DYNMATVERBOSE) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -rf $(OBJECTS) $(BIN) *~

VocabTree_test.o: VocabTree_test.cpp
	$(CXX) $(VTREEVERBOSE) $(DYNMATVERBOSE) $(CXXFLAGS) -c $< -o $@
