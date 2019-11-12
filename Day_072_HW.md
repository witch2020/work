

```python
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
#Sigmoid 數學函數表示方式
def sigmoid(x):
    return  (1 / (1 + np.exp(-x)))
def dsigmoid(x):
    return (x * (1 - x))

x = linspace(10, -10, 100)
plt.plot(x, sigmoid(x), 'b', label = 'linspace(10, -10, 100)')
plt.grid()
plt.title('Sigmoid Function')
plt.text(4, 0.8, r'$\sigma(x) = \frac{1}{1 + e^{-x}}$', fontsize = 18)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))

plt.show()
```


![png](output_1_0.png)


補充: 對使用反向傳播訓練的類神經網絡來說，梯度的問題是最重要的，使用 sigmoid函數容易發生梯度消失問題，是類神經網絡加深時主要的訓練障礙。原因是這當函數在接近飽和區求導後趨近於0，也就是所謂梯度消失，會造成更新的訊息無法藉由反向傳播傳遞。


```python
import numpy as np
from numpy import *
import matplotlib.pylab as plt
%matplotlib inline

#Softmax 數學函數表示方式
def softmax(x):
     return np.exp(x) / float(sum(np.exp(x)))

x = plt.linspace(10,-10,100)
plt.grid()
plt.title('Softmax Function')
plt.text(0.5, 0.1, r'$\S(x) = \frac{e^{x}}{sum(e^{x})}$', fontsize = 18)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

#列印所有Softmax 值並輸出成一陣列
print(softmax(x))
plt.plot(x, softmax(x), 'r')
plt.show()
```

    [1.82921579e-01 1.49461275e-01 1.22121583e-01 9.97829100e-02
     8.15304625e-02 6.66167816e-02 5.44311347e-02 4.44745056e-02
     3.63391588e-02 2.96919425e-02 2.42606455e-02 1.98228499e-02
     1.61968229e-02 1.32340745e-02 1.08132767e-02 8.83529506e-03
     7.21912894e-03 5.89859447e-03 4.81961426e-03 3.93800281e-03
     3.21765712e-03 2.62907820e-03 2.14816306e-03 1.75521768e-03
     1.43415049e-03 1.17181342e-03 9.57463460e-04 7.82322732e-04
     6.39219023e-04 5.22292070e-04 4.26753580e-04 3.48691141e-04
     2.84908007e-04 2.32792185e-04 1.90209471e-04 1.55416054e-04
     1.26987104e-04 1.03758422e-04 8.47787680e-05 6.92709019e-05
     5.65997592e-05 4.62464418e-05 3.77869697e-05 3.08749175e-05
     2.52272289e-05 2.06126243e-05 1.68421305e-05 1.37613414e-05
     1.12440951e-05 9.18730750e-06 7.50675071e-06 6.13360401e-06
     5.01163548e-06 4.09489921e-06 3.34585378e-06 2.73382492e-06
     2.23374935e-06 1.82514839e-06 1.49128937e-06 1.21850036e-06
     9.95610352e-07 8.13491735e-07 6.64686542e-07 5.43101031e-07
     4.43756133e-07 3.62583560e-07 2.96259203e-07 2.42067002e-07
     1.97787724e-07 1.61608081e-07 1.32046476e-07 1.07892326e-07
     8.81564912e-08 7.20307667e-08 5.88547851e-08 4.80889749e-08
     3.92924637e-08 3.21050242e-08 2.62323225e-08 2.14338646e-08
     1.75131483e-08 1.43096155e-08 1.16920781e-08 9.55334469e-09
     7.80583179e-09 6.37797672e-09 5.21130715e-09 4.25804662e-09
     3.47915801e-09 2.84274493e-09 2.32274554e-09 1.89786526e-09
     1.55070475e-09 1.26704739e-09 1.03527708e-09 8.45902561e-10
     6.91168729e-10 5.64739054e-10 4.61436094e-10 3.77029476e-10]
    


![png](output_3_1.png)


# 作業 :
## 寫出 ReLU & dReLU 一階導數並列印
Rectified Linear Unit- Relu 
f(x)=max(0,x)


```python
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# ReLU函數圖形
x =  np.arange(-10,10,0.01)
y = list(map(lambda x:x if x>0 else 0,x))
plt.title('ReLU function')
plt.plot(x, y)
plt.grid(color='r', linewidth='0.2', linestyle='--')
```


![png](output_6_0.png)



```python
# ReLU導函數圖形
deri_y =list(map(lambda x: 1 if x>0 else 0,x))
plt.title('dReLU Function')
plt.plot(x,deri_y)
plt.grid(color='r', linewidth='0.2', linestyle='--')
```


![png](output_7_0.png)



```python
#ReLU 數學函數表示方式
def Relu(x):
    return abs(x) * (x > 0)
def dRelu(x):
    return (1 * (x > 0))

x = linspace(-10,10,100)
plt.plot(x, Relu(x), 'r')
plt.plot(x, dRelu(x), 'b')
plt.grid()
plt.title('ReLU & dReLU')
plt.text(0, 9, r'$f(x) = (abs(x) * (x > 0))$', fontsize = 15)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

#列印所有 Relu 值並輸出成一陣列
print(Relu(x))
plt.plot(x, Relu(x), 'r')
plt.show()
```

    [ 0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.          0.          0.          0.
      0.          0.          0.1010101   0.3030303   0.50505051  0.70707071
      0.90909091  1.11111111  1.31313131  1.51515152  1.71717172  1.91919192
      2.12121212  2.32323232  2.52525253  2.72727273  2.92929293  3.13131313
      3.33333333  3.53535354  3.73737374  3.93939394  4.14141414  4.34343434
      4.54545455  4.74747475  4.94949495  5.15151515  5.35353535  5.55555556
      5.75757576  5.95959596  6.16161616  6.36363636  6.56565657  6.76767677
      6.96969697  7.17171717  7.37373737  7.57575758  7.77777778  7.97979798
      8.18181818  8.38383838  8.58585859  8.78787879  8.98989899  9.19191919
      9.39393939  9.5959596   9.7979798  10.        ]
    


![png](output_8_1.png)


補充: ReLU的分段線性性質能有效的克服梯度消失的問題，且Relu會使部分神經元的輸出為0，讓神經網路變得稀疏，緩解過度擬合的問題，因此在深度學習領域 Relu 激勵函數目前為主流。不過ReLU有個問題是如果把一個神經元停止後，就無法開啟造成Dead ReLU Problem，因此又有 Leaky ReLU / Randomized Leaky ReLU (x < 0 時取一個微小值而非 0 ) 等代替的方法。

![](relufamily.jpg)

![image.png](attachment:image.png)


```python

```
