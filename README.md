# 文件说明

raw_array.h: 包含对数组进行计算的各种函数

raw_matrix.h: 包含对矩阵进行计算的各种函数

tensor.h: 定义了tensor，定义了对tensor的各种计算，实现了tensor反向传播过程

test.c: 各种测试文件

test.py: 用pytorch实现的test.c的功能，用来验证test.c的计算结果是否正确

# 编译

编译环境：

Archlinux 2022-08-06

gcc 12.1.1

make 4.3

在项目根目录执行`make`即可生成tensor的动态链接库

执行`make test_mnist_cpp`生成一个在mnist数据集上的全连接神经网络示例程序

**运行**

将libtensor.so所在文件夹的绝对路径添加至ld.so.conf的最后一行：

`sudo nano /etc/ld.so.conf`

重新加载系统的动态链接库：

`sudo ldconfig -v`

运行生成的可执行文件：

`./test_mnist`