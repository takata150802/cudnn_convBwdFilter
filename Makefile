CXX = g++
CFLAGS = -std=c++11 -O3 -Wall
INCDIR = -I/usr/include \
         -I/usr/local/cuda/include
LDFLAGS = 
OBJS = main.o run_conv_bwd_filter.o
NVOBJS = /usr/lib/x86_64-linux-gnu/libcudnn.so \
         /usr/local/cuda/lib64/libcudart.so
NAME = main.out

$(NAME): $(OBJS)
	$(CXX) -o $(NAME) $(OBJS) $(NVOBJS)

main.o: main.cc
	$(CXX) $(INCDIR) $(LDFLAGS) $(CFLAGS) -c $<

run_conv_bwd_filter.o: run_conv_bwd_filter.cc
	$(CXX) $(INCDIR) $(LDFLAGS) $(CFLAGS) -c $<
clean:
	$(RM) $(NAME) $(OBJS)
