import numpy


#N维数组对象ndarray,同类型数据的集合，以0下标开始索引
#创建一个 ndarray 只需调用 NumPy 的 array 函数即可,ndarray 对象由计算机内存的连续一维部分组成，并结合索引模式，将每个元素映射到内存块中的一个位置。内存块以行顺序(C样式)或列顺序(FORTRAN或MatLab风格，即前述的F样式)来保存元素。
# 参数:object数组或嵌套的数列，dtype数组元素的数据类型，可选，copy对象是否需要复制，可选，order创建数组的样式，C为行方向，F为列方向，A为任意方向，subok默认返回一个与基本类型一致的数组，ndmin指定生成数组的最小维度
a = numpy.array([1,2,3])
print(a)
#多于一个维度
b = numpy.array([[1,2,3],[4,5,6]])
print(b)
#最小维度
c = numpy.array([1, 2,3,4,5],ndmin=2)
print(c)
#dtype参数
d = numpy.array([1,2,3],dtype=complex)
print(d)
#数据类型对象是用来描述与数组对应的内存区域如何使用
#使用标量类型
dt1 = numpy.dtype(numpy.int)
print(dt1)
#int8,int16,int32,int64四种数据类型可以使用字符串‘i1’'i2''i4''i8'代替
dt2 = numpy.dtype('i2')
print(dt2)
#创建结构化数据类型
dt3 = numpy.dtype([('age',numpy.int8)])
print(dt3)
#将数据类型应用于ndarray对象
dt4 = numpy.dtype([('age',numpy.int8)])
a1 = numpy.array([(10,),(20,),(30,)], dtype= dt4)
print(a1)
#类型字段名可以用于存取实际的age列
print(a1['age'])
#定义一个结构化数据类型 student，包含字符串字段 name，整数字段 age，及浮点字段 marks，并将这个 dtype 应用到 ndarray 对象。
student = numpy.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)
b1 = numpy.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student)#使用序列设置数组元素
print(b1)
print(b1['name'])
# empty创建随机数数组，zeros都是0的数组，ones都是1的数组
#numpy.empty 方法用来创建一个指定形状（shape）、数据类型（dtype）且未初始化的数组
#numpy.empty(shape, dtype = float, order = 'C')order	有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序.
x = numpy.empty((3,2,1), dtype = numpy.int8)
print(x)
z = numpy.zeros(5)
print(z)
#dtype还可以自定义类型,shape数组形状(n,m)n行m列
z1 = numpy.zeros((2,2),dtype=[('x','i4'),('y','i4')])
print(z1)
#ndarray.ndim	秩，即轴的数量或维度的数量
print(x.ndim)
#ndarray.shape	数组的维度，对于矩阵，n 行 m 列
print(x.shape)
#ndarray.size	数组元素的总个数，相当于 .shape 中 n*m 的值
print(x.size)
#ndarray.dtype	ndarray 对象的元素类型
print(x.dtype)
#ndarray.real	ndarray元素的实部
print(x.real)
#ndarray.imag	ndarray 元素的虚部
print(x.imag)
#ndarray.flags	ndarray 对象的内存信息.C_CONTIGUOUS (C)	数据是在一个单一的C风格的连续段中
#F_CONTIGUOUS (F)	数据是在一个单一的Fortran风格的连续段中
#OWNDATA (O)	数组拥有它所使用的内存或从另一个对象中借用它
#WRITEABLE (W)	数据区域可以被写入，将该值设置为 False，则数据为只读
#ALIGNED (A)	数据和所有元素都适当地对齐到硬件上
#UPDATEIFCOPY (U)	这个数组是其它数组的一个副本，当这个数组被释放时，原数组的内容将被更新
print(x.flags)
#reshape函数来调整数组大小
x1 = numpy.reshape(3,1,2)
print(x1)
#从已有的数组创建数组numpy.asarray(a, dtype = None, order = None)
#a	任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组
y1 = [1,2,3]
y1_1 = numpy.asarray(y1)#将列表转换为ndarray
print(y1_1)
#实现动态数组:numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
#buffer	可以是任意对象，会以流的形式读入。count	读取的数据数量，默认为-1，读取所有数据。offset	读取的起始位置，默认为0
s = b'Hello World'
s2 = numpy.frombuffer(s,dtype='S1')
print(s2)
#从可迭代对象中建立 ndarray 对象，返回一维数组。numpy.fromiter(iterable, dtype, count=-1)
#iterable	可迭代对象 count	读取的数据数量，默认为-1，读取所有数据
list = range(5)
it = iter(list)
s3 = numpy.fromiter(it, dtype=float)
print(s3)
#从数值范围创建数组numpy.arange(start, stop, step, dtype)
#start	起始值，默认为0. stop	终止值（不包含）. step	步长，默认为1. dtype	返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。
s4 = numpy.arange(10,20,2)
print(s4)
#创建一个一维数组，数组是一个等差数列构成的np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#num要生成的等步长的样本数量，默认为50。endpoint该值为 ture 时，数列中中包含stop值，反之不包含，默认是True。retstep	如果为 True 时，生成的数组中会显示间距，反之不显示。
s5 = numpy.linspace(1,10,9)
print(s5)
#创建一个于等比数列np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
#base对数 log 的底数。
s6 = numpy.logspace(1.0, 2.0, 10,base = 2)
print(s6)
#ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。
s7 = slice(2,7,2)
print(s6[s7])
#或者
s8 = s6[2:7:2]
print(s8)
#二维数组索引
s9 = numpy.array([[1,2,3],[3,4,5],[6,7,8]])
print(s9[...,1])#取第二列。索引从0开始
print(s9[1,...])#取第二行
print(s9[...,1:])#取第二、第三列
#NumPy 比一般的 Python 序列提供更多的索引方式。除了之前看到的用整数和切片的索引外，数组可以由整数数组索引、布尔索引及花式索引。
s10 = numpy.array([[1,2],[3,4],[5,6]])
print(s10[[0,1,2],[0,1,0]])##[0,1,2]代表行数，[0,1,0]代表列数，即取坐标为(0,0),(1,1),(2,0)的数
#广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。
a = numpy.array([1,2,3,4])
b = numpy.array([10,20,30,40])
c = a * b
print (c)
#当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制。列数相同，行数相同或行数=1 的数组可加减乘除.若条件不满足，抛出 "ValueError: frames are not aligned" 异常。
a = numpy.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = numpy.array([1,2,3])
print(a + b)#4x3 的二维数组与长为 3 的一维数组相加，等效于把数组 b 在二维上重复 4 次再运算
#NumPy 迭代器对象 numpy.nditer 提供了一种灵活访问一个或者多个数组元素的方式。
#按单个元素迭代
a = numpy.arange(0,60,5).reshape(4,3)
print(a)
for x in numpy.nditer(a):
    print(x,end = ',')
print("\n")
#按一维数组迭代
for y in numpy.nditer(a,flags = ['external_loop'],order = 'F'):#Fortran order，即是列序优先.C order，即是行序优先.
    print(y)
for y in numpy.nditer(a, flags = ["external_loop"],order = 'C'):#external_loop	给出的值是具有多个值的一维数组，而不是零维数组
    print(y)
#两个数组同时迭代
a = numpy.arange(0,60,5).reshape(4,3)
b = numpy.array([1,2,3])
for x,y in numpy.nditer([a,b]):
    print(x,":",y,end='|')
#迭代过程中可以对元素进行运算
print("\n")
for x in numpy.nditer(a,op_flags=['readwrite']):
    x[...] = 2*x
    print(x,end=",")
print("\n")
#多维数组转一维.
# flatten返回一份数组拷贝，对拷贝所做的修改不会影响原始数组。flat数组元素迭代器。ravel返回展开数组。
a = numpy.arange(8).reshape(2,4)
print(a)
print(a.flatten())
#数组转置
#numpy.transpose(arr, axes)arr：要操作的数组。axes：整数列表，对应维度，通常所有维度都会对换。
print(numpy.transpose(a))
#翻转数组。
#数组按轴翻转 transpose	对换数组的维度。darray.T和 self.transpose() 相同。rollaxis向后滚动指定的轴。swapaxes对换数组的两个轴
b = numpy.arange(8).reshape(2,2,2)
print(b)
#numpy.rollaxis(arr, axis, start)axis：要向后滚动的轴，其它轴的相对位置不会改变。start：默认为零，表示完整的滚动。会滚动到特定位置。
print(numpy.rollaxis(b ,2,0))#轴滚动规则：4=a[1][0][0]  --->  4=a[0][1][0]
#交换数组的两个轴numpy.swapaxes(arr, axis1, axis2)axis1：对应第一个轴的整数。axis2：对应第二个轴的整数。
print(numpy.swapaxes(b,2,0)) #轴交换规则：1=a[0][0][1]  --->  1=a[1][0][0]把第3个轴和第1个轴交换
#删除一维条目
c = numpy.squeeze(b)
print(c)
#数组连接
a = numpy.array([[1,2],[3,4]])
b = numpy.array([[5,6],[7,8]])
print(numpy.concatenate((a,b)))#！ 两个数组的维度必须相同
print(numpy.concatenate((a,b),axis=1))#延轴1连接
#数组分割
a = numpy.arange(8)
#必须按等分分割
b = numpy.split(a,4)#将一个数组分割为多个子数组,默认为0，横向切分。为1时，纵向切分
#可以分割出指定的数组
c = numpy.split(a,[4,7])
#数组元素的添加与删除
#resize	返回指定形状的新数组numpy.resize(arr, shape)
#append	将值添加到数组末尾numpy.append(arr, values, axis=None)
#insert	沿指定轴将值插入到指定下标之前numpy.insert(arr, obj, values, axis)obj：在其之前插入值的索引.
#delete	删掉某个轴的子数组，并返回删除后的新数组numpy.delete(arr, obj, axis)
#unique	查找数组内的唯一元素,用于去除数组中的重复元素.numpy.unique(arr, return_index, return_inverse, return_counts)
    #return_index：如果为true，返回新列表元素在旧列表中的位置（下标），并以列表形式储.
    #return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式储
    #return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数
a = numpy.array([5,2,6,2,7,5,6,8,2,9])
#去重结果：
u = numpy.unique(a)
print(u)
#去重数组的索引数组：
u,indices = numpy.unique(a, return_index = True)
print(indices)
#对字段排序
dt = numpy.dtype([('name','S10'),('age',int)])
a = numpy.array([("raju",21),("anil",25),("ravi",17),("amar",27)],dtype=dt)
dtt = numpy.sort(a,order= "age")#指定需要排序的字段
print(dtt)
#按条件查找1
a = numpy.array([1.0,5.55,123,0.567,25,232])
#根据条件返回索引
t1 = numpy.where(a>10)
print(t1)
#根据索引返回值
print(a[numpy.where(a>10)])
#直接返回值
print(numpy.extract(a>10,a))
#按条件查找2
#条件数组
x = numpy.array([[0,1,2],[3,4,5],[6,7,8],[9,1,0]])
print(x)
#待查找的函数
y = numpy.random.rand(4,3)
print(y)
#简单查找
print(y[x>5])#等于np.extract(x>5,y),这时[x>5]是个坐标号
#！多个并列条件查找，条件之间用&
print(y[(x>5) & (x%2==0)])
#！多个可选条件查找，条件之间用|
print(y[(x>5) | (x%2==0)] )
#数组的算数函数和运算.NumPy 包含大量的各种数学运算的函数，包括三角函数，算术运算的函数，复数处理函数等。
#标准的三角函数：sin()、cos()、tan()。
#包含简单的加减乘除: add()，subtract()，multiply() 和 divide()。
#numpy.around() 函数返回指定数字的四舍五入值numpy.around(a,decimals)decimals: 舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
#numpy.floor() 返回数字的下舍整数。
#numpy.ceil() 返回数字的上入整数。
#求数组中每个元素的倒数
a = numpy.array([0.25,  1.33,  1,  100])
print(numpy.reciprocal(a))
#！对任何大于1的整数求倒数，结果都为0
#数组幂运算power()
#数组求余数remainder
a = numpy.array([10,20,30])
b = numpy.array([3,5,7])
print(numpy.remainder(a,b))
#矩阵积
A = numpy.arange(0,9).reshape(3,3)
B = numpy.ones((3,3))
print(A*B)
#矩阵的乘法
print(A.dot(B))
print(B.dot(A))
#矩阵的自增自减
A += 1  #减法同理
A *= 10
#聚合函数
print(A.sum(),A.max(),A.mean(),A.std())
#数组的统计函数
a = numpy.array([[3,7,5],[8,4,3],[2,4,9]])
print(numpy.amin(a))#最大值amax()，不指定轴就是对所有值求最小值
print(numpy.amin(a,axis = 0))#指定0轴，就是按列取最小值
print(numpy.ptp(a))#求值范围(最大值 - 最小值)，即计算（9-2）
print(numpy.ptp(a,axis=0))#计算（8-2），（7-4），（9-3）
print(numpy.percentile(a,50))#求百分位数.计算中位数，可加轴数[axis = ]
print(numpy.median(a))#求中值,可加轴数[axis = ]
print(numpy.mean(a))#求算数平均值.可加轴数[axis = ]
a = numpy.array([1,2,3,4])
#求加权平均值
wts = numpy.array([8,7,6,5]) #设定权重函数，必须与数组函数同形状，不设定则默认为求平均值
numpy.average(a,weights=wts)#计算过程：(1*8+2*7+3*6+4*5)/(8+7+6+5)
print(numpy.var(a))#方差
print(numpy.std(a))#标准差




