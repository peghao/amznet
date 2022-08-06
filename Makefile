cc = gcc

# 生成动态链接库
tensor:
	$(cc) -c raw_array.c
	$(cc) -c raw_matrix.c
	$(cc) -c linked_list.c
	$(cc) -c tensor.c
	$(cc) -shared -fPIC tensor.c raw_matrix.c raw_array.c linked_list.c -o libtensor.so -lm

test_mnist:tensor
	gcc test_mnist.c -o test_mnist -L ./ -ltensor

# 编译并链接
test_mnist_cpp:tensor
	 g++ test_mnist.cpp model.cpp -o test_mnist -L. -ltensor

clean:
	rm *.o *.so