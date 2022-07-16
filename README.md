# 文件说明

raw_array.h: 包含对数组进行计算的各种函数

raw_matrix.h: 包含对矩阵进行计算的各种函数

tensor.h: 定义了tensor，定义了对tensor的各种计算，实现了tensor反向传播过程

test.c: 各种测试文件

test.py: 用pytorch实现的test.c的功能，用来验证test.c的计算结果是否正确

# 编译

*（以编译test_sum_all.c为例）*

**Windows编译：**
编译器：clang version 14.0.0 Target: x86_64-w64-windows-gnu（下载链接：[LLVM-MingW](https://github.com/mstorsjo/llvm-mingw/)）

`clang test_sum_all.c`

编译生成的文件是a.exe

**Ubuntu编译：**

编译器：gcc version 9.4.0

`gcc test_sum_all.c -lm`

编译器：clang version 10.0.0-4ubuntu1 Target: x86_64-pc-linux-gnu

`clang test_sum_all.c -lm`

编译生成的文件是a.out