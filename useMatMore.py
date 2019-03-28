import numpy as np
import matplotlib.pyplot as plt

#Matplotlib默认情况下不支持中文，可以去菜鸟教程下载字体：https://www.fontpalace.com/font-details/SimHei/
#再将下载好的SimHei.ttf 文件放在当前执行的代码文件中。
#计算正弦和余弦曲线上的点的x和y坐标
x = np.arange(0,3 *np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
#建立subplot网格，高为2，宽为1
plt.subplot(2,1,1)
#绘制第一个图像
plt.plot(x,y_sin)
plt.title('Sine')
#将第二个subplot激活，并绘制第二个图像
plt.subplot(2,1,2)
#绘制第二个图像
plt.plot(x,y_cos)
plt.title('Cosine')
#展示图像
plt.show()