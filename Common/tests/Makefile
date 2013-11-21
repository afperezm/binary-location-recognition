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

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLES = $(OBJECTS:.o=)

all: $(EXECUTABLES)

$(EXECUTABLES): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $@.o $(LDFLAGS) -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJECTS) $(EXECUTABLES) *.log *~

