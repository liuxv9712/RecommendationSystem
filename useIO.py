import numpy

a = numpy.array([1, 2, 3, 4, 5])
# 保存为npy文件
numpy.save('outfile', a)
# 读取npy文件
b = numpy.load('outfile.npy')
print(b)

# savetxt() 函数是以简单的文本文件格式存储数据，对应的使用 loadtxt() 函数来获取数据。
c = numpy.array([1, 2, 3, 4, 5, 6])
# 保存为txt文件
numpy.savetxt('out.txt', c, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔
# 读取txt文件。load 时也要指定为逗号分隔
d = numpy.loadtxt('out.txt')
print(d)
# numpy.savez() 函数将多个数组保存到以 npz 为扩展名的文件中。numpy.savez(file, *args, **kwds)
# file：要保存的文件.args: 要保存的数组.kwds: 要保存的数组使用关键字名称。

# Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化。
