# Makefile for Incremental K-Means Tests

CXXFLAGS = -O2 $(DEBUGFLAGS) -fmessage-length=0 -std=c++0x
LDFLAGS = -L../../lib/ -lboost_iostreams

# IncrementalKMeans
CXXFLAGS += -I../../IncrementalKMeansLib/include
LDFLAGS += -lincrementalkmeans

# Common
CXXFLAGS += -I../../Common/include/
LDFLAGS += -lcommon

# KMajority
CXXFLAGS += -I../../KMajorityLib/include
LDFLAGS += -lkmajority

# OpenCV
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

# GoogleTest
CXXFLAGS += -Wextra -pthread
#LDFLAGS += -L/home/andresf/workspace-cpp/gtest-1.7.0/make
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
