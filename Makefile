CXX = g++
CFLAGS = -lm -pthread -Ofast -Wall -funroll-loops

all: doc2label predict

doc2label: doc2label.cc
	$(CXX) doc2label.cc -o doc2label $(CFLAGS)

predict: classify.cc
	$(CXX) classify.cc -o predict $(CFLAGS)

clean:
	rm -rf doc2label predict
