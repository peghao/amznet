tensor:linked_list.c raw_array.c raw_matrix.c tensor.c
	gcc -c raw_array.c
	gcc -c raw_matrix.c
	gcc -c linked_list.c
	gcc -c tensor.c
	gcc -shared -fPIC tensor.c raw_matrix.c raw_array.c linked_list.c -o libtensor.so -lm

test_mnist:tensor
	gcc test_mnist.c -o test_mnist -L ./ -ltensor

test_mnist_cpp:tensor
	 g++ test_mnist.cpp model.cpp -o test_mnist -L. -ltensor

clean:
	rm *.o *.so