# Makefile for AGAST and DBRIEF Tests

CXXFLAGS = -O2 $(DEBUGFLAGS) -fmessage-length=0 -std=c++0x
LDFLAGS = -L../../lib/ -lboost_iostreams

# Common
CXXFLAGS += -I../../Common/include/
LDFLAGS += -lcommon

# OpenCV Extensions
CXXFLAGS += -I../include
LDFLAGS += -lopencv_extensions

# DBrief
CXXFLAGS += -I../../../dbrief_lib
LDFLAGS += -L../../../dbrief_lib/lib -ldbrief

# Agast
CXXFLAGS += -I../../../agast_lib
LDFLAGS += -L../../../agast_lib/lib -lagast

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
	rm -rf $(OBJECTS) $(EXECUTABLES) test_*.yaml.gz *~
