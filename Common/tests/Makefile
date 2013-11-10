# Makefile for Common Tests

CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++0x
LDFLAGS = -L../../lib/

# Common
CXXFLAGS += -I../include
LDFLAGS += -lcommon

# OpenCV
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

# GoogleTest
CXXFLAGS += -Wextra -pthread
LDFLAGS += -L/home/andresf/workspace-cpp/gtest-1.7.0/make
LDFLAGS += -lgtest_main -lpthread

SOURCES = DynamicMat_test.cpp
OBJECTS = $(SOURCES:.cpp=.o)

BIN =  DynamicMat_test

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -rf $(OBJECTS) $(BIN) *~

DynamicMat_test.o: DynamicMat_test.cpp
	$(CXX) $(DYNMATVERBOSE) $(CXXFLAGS) -c $< -o $@
