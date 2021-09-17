import numpy as np

#print(np.zeros_like(np.array([1.0,2.0])))

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)#xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val =x[idx]
        #f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val#値を元に戻す

    return grad

def function_2(x):#目的関数の定義
    return x[0]**2 + x[1]**2


print(numerical_gradient(function_2, np.array([3.0,4.0])))
