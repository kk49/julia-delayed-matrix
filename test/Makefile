all: libtestcuda.so

testcuda.o: testcuda.c
	gcc -std=c99 -c -fPIC testcuda.c -o testcuda.o

libtestcuda.so: testcuda.o
	gcc -shared -Wl,-soname,libtestcuda.so -o libtestcuda.so testcuda.o
