CXX = g++

CXXFLAGS = -O2 -g -Wall -fmessage-length=0 -std=c++11 -I./include/
LDFLAGS =

# DBrief
CXXFLAGS += -I../dbrief_lib
LDFLAGS += -L../dbrief_lib/lib -ldbrief

# Agast
CXXFLAGS += -I../agast_lib
LDFLAGS += -L../agast_lib/lib -lagast

# DBow
CXXFLAGS += -I../DBoW2/DUtils -I../DBoW2/DUtilsCV -I../DBoW2/DVision -I../DBoW2/DBoW2
LDFLAGS += -L../DBoW2/lib -lDBoW2 -lDUtils -lDUtilsCV -lDVision -lstdc++

# OpenCV (this goes last: beware of the linking order)
CXXFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

LDFLAGS += -Wl,-rpath=/opt/ros/groovy/lib
LDFLAGS += -Wl,-rpath=../agast_lib/lib
LDFLAGS += -Wl,-rpath=../dbrief_lib/lib
LDFLAGS += -Wl,-rpath=../DBoW2/lib

SOURCES=$(wildcard src/*.cpp)

OBJS += $(SOURCES:.cpp=.o)

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJS) $(TARGET) *~

cleanObjs:
#	find ./ -name "*.o" | xargs -I {} rm -f {}
	rm $(OBJS)
