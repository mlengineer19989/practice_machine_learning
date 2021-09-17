import sys, os
sys.path.append(os.pardir)#親ディレクトリーのファイルのをインポートするための設定
from dataset.mnist import load_mnist
import numpy as np

#最初の呼び出しは数分待ちます
(x_train,t_train), (x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

#それぞれのデータの形状を出力
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(batch_mask)