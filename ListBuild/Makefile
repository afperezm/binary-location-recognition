# Makefile for ListBuild

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

BIN = ListBuild

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CXX) -o $(CXXFLAGS) -o $(BIN) $(OBJECTS) $(LDFLAGS)

clean:
	rm -rf $(OBJECTS) $(BIN) *~
