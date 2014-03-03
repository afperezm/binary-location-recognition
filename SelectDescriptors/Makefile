# Makefile for SelectDescriptors

CXXFLAGS = -O2 $(DEBUGFLAGS) -fmessage-length=0 -std=c++0x
LDFLAGS = -L../lib/

# Common
CXXFLAGS += -I../Common/include/
LDFLAGS += -lcommon

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

BIN = SelectDescriptors

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(BIN) $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(BIN) *~
