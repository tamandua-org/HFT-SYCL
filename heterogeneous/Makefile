CXX=icpx #-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64
CXXFLAGS=-fsycl -I.
#CXXFLAGS= -I.
LIBS=

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o