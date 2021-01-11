import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

# 数据加载与划分
data = pd.read_csv('F:/PycharmProjects/stocks_predict/data_stocks.csv')
data.info()

data.drop('DATE', axis=1, inplace=True)
data_train = data.iloc[:int(data.shape[0] * 0.8), :]
data_test = data.iloc[int(data.shape[0] * 0.8):, :]
print(data.head())

scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scale.fit(data_train)
data_train = scale.transform(data_train)
data_test = scale.transform(data_test)

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

# 创建一个顺序模型
model = Sequential()

model.add(Dense(units=hidden_1, activation='sigmoid', input_shape=(num_input,)))
model.add(Dense(units=hidden_2, activation='relu'))
model.add(Dense(units=hidden_3, activation='relu'))
model.add(Dense(units=hidden_4, activation='relu'))
model.add(Dense(units=num_output, activation='sigmoid'))

model.summary()

# 使用优化器编译模型
model.compile(loss='mean_squared_error',
              optimizer='adam')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
scores = model.evaluate(X_test, y_test)
print('\n loss: ', scores)

# 进行预测并绘制图像
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
plt.plot(y_test, label='test')
plt.plot(y_test_predict, label='predict')
plt.title('Keras predict stocks')
plt.legend()
plt.show()
