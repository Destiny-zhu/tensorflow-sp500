import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd

# 数据预览
data = pd.read_csv('F:/PycharmProjects/stocks_predict/data_stocks.csv')
data.info()
print(data.head())
plt.plot(data['SP500'])
plt.show()

# 数据划分
data.drop('DATE', axis=1, inplace=True)
data_train = data.iloc[:int(data.shape[0] * 0.8), :]
data_test = data.iloc[int(data.shape[0] * 0.8):, :]

# 数据归一化
scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scale.fit(data_train)
data_train = scale.transform(data_train)
data_test = scale.transform(data_test)

# 划分X，Y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# 设置一些超参数
num_input = X_train.shape[1]
num_output = 1
hidden_1 = 1024
hidden_2 = 512
hidden_3 = 256
hidden_4 = 128
batch_size = 256
epochs = 10

# 设置两个占位符
X = tf.placeholder(shape=[None, num_input], dtype=tf.float32)
Y = tf.placeholder(shape=[None], dtype=tf.float32)

# 第一层
W1 = tf.get_variable('W1', [num_input, hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable('b1', [hidden_1], initializer=tf.zeros_initializer())
# 第二层
W2 = tf.get_variable('W2', [hidden_1, hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable('b2', [hidden_2], initializer=tf.zeros_initializer())
# 第三层
W3 = tf.get_variable('W3', [hidden_2, hidden_3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable('b3', [hidden_3], initializer=tf.zeros_initializer())
# 第四层
W4 = tf.get_variable('W4', [hidden_3, hidden_4], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b4 = tf.get_variable('b4', [hidden_4], initializer=tf.zeros_initializer())
# 输出层
W5 = tf.get_variable('W5', [hidden_4, num_output], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b5 = tf.get_variable('b5', [num_output], initializer=tf.zeros_initializer())

# 设置网络体系结构
h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
h3 = tf.nn.relu(tf.add(tf.matmul(h2, W3), b3))
h4 = tf.nn.relu(tf.add(tf.matmul(h3, W4), b4))
out = tf.transpose(tf.add(tf.matmul(h4, W5), b5))

# 设置损失函数（loss function）和优化器（Optimizer）
loss = tf.reduce_mean(tf.squared_difference(out, Y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 训练过程
with tf.Session() as tfs:
    # 初始化所有变量
    tfs.run(tf.global_variables_initializer())

    for e in range(epochs):
        # 将数据打乱
        shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        for i in range(y_train.shape[0] // batch_size):
            start = i * batch_size
            batch_x = X_train[start : start + batch_size]
            batch_y = y_train[start : start + batch_size]
            tfs.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            if i % 50 == 0:
                print('MSE Train:', tfs.run(loss, feed_dict={X: X_train, Y: y_train}))
                print('MSE Test:', tfs.run(loss, feed_dict={X: X_test, Y: y_test}))
                y_predict = tfs.run(out, feed_dict={X: X_test})
                y_predict = np.squeeze(y_predict)

    plt.plot(y_test, label='test')
    plt.plot(y_predict, label='predict')
    plt.title('TensorFlow predict stocks')
    plt.legend()
    plt.show()
