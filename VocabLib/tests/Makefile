# Makefile for VocabLib Tests

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++0x
LDFLAGS = -L../../lib/ -lboost_iostreams

# KMajority
CXXFLAGS += -I../../KMajorityLib/include
LDFLAGS += -lkmajority

# VocabLib
CXXFLAGS += -I../include
LDFLAGS += -lvocab

# Common
CXXFLAGS += -I../../Common/include/
LDFLAGS += -lcommon

# OpenCV
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

# GoogleTest
CXXFLAGS += -Wextra -pthread
LDFLAGS += -L/home/andresf/workspace-cpp/gtest-1.7.0/make
LDFLAGS += -lgtest_main -lpthread

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLES = $(OBJECTS:.o=)

#DEBUG = -DDEBUG
#VTREEVERBOSE = -DVTREEVERBOSE

all: $(EXECUTABLES)

$(EXECUTABLES): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $@.o $(LDFLAGS) -o $@

.cpp.o:
	$(CXX) $(VTREEVERBOSE) $(DEBUG) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJECTS) $(EXECUTABLES) test_tree.yaml.gz test_idf.yaml.gz test_di.yaml.gz *~
