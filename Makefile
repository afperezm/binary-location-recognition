CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		MediaEval-PlacingTask.o

LIBS =

TARGET =	MediaEval-PlacingTask

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
