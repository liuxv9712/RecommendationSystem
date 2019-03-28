import numpy as np
from matplotlib import pyplot as plt

#设定x和y
x = np.arange(1,11)
y = x**2+5
#打印标题
plt.title("Matplotlib demo")
#打印x轴和y轴标签
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
#设置线段的格式
plt.plot(x,y,'m')
#输出图表
plt.show()