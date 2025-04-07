import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, LeakyReLU, Input, Flatten
from keras.optimizers import Adam

# 步骤 1: 加载并预处理数据
file_path = 'initial_data.xlsx'
df = pd.read_excel(file_path)

# 时间序列数据，只取第2列
time_series_data = df.iloc[:, 1].values

# 归一化数据到[0, 1]区间（为了更好地训练GAN）
scaler = MinMaxScaler(feature_range=(-1, 1))
time_series_data = scaler.fit_transform(time_series_data.reshape(-1, 1))

# 将数据划分为训练样本，假设使用过去n个数据点预测下一个数据点
def create_dataset(data, time_step=50):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 50
X, y = create_dataset(time_series_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # 适应LSTM输入的形状

# 步骤 2: 定义GAN模型

# 生成器模型
def build_generator():
    model = Sequential()
    model.add(LSTM(100, input_shape=(50, 1), return_sequences=True))
    model.add(LSTM(100, return_sequences=True))  # 保持时间步长
    model.add(Dense(1, activation='tanh'))  # 输出一个数值，适应时间序列数据
    return model

# 判别器模型
def build_discriminator():
    model = Sequential()
    model.add(LSTM(100, input_shape=(50, 1), return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))  # 输出0或1，0表示假，1表示真
    return model

# GAN模型（将生成器和判别器组合）
def build_gan(generator, discriminator):
    discriminator.trainable = False  # 在GAN训练时，我们只训练生成器
    noise = Input(shape=(50, 1))  # 输入噪声
    generated_sequence = generator(noise)  # 生成器输出
    validity = discriminator(generated_sequence)  # 判别器评估生成的序列
    gan = Model(noise, validity)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# 创建生成器、判别器和GAN模型
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)

# 步骤 3: 训练GAN
def train_gan(epochs, batch_size, X_train):
    for epoch in range(epochs):
        # 随机选择一批真实样本
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_sequences = X_train[idx]

        # 生成一批假的时间序列
        noise = np.random.normal(0, 1, (batch_size, 50, 1))  # 随机噪声
        fake_sequences = generator.predict(noise)

        # 判别器训练
        d_loss_real = discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))  # 真实数据标记为1
        d_loss_fake = discriminator.train_on_batch(fake_sequences, np.zeros((batch_size, 1)))  # 假数据标记为0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # 计算总损失

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, 50, 1))  # 随机噪声
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # 生成器的目标是让判别器误判

        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

# 训练GAN
train_gan(epochs=10000, batch_size=64, X_train=X)

# 步骤 4: 使用训练好的生成器生成新的时序数据
noise = np.random.normal(0, 1, (1, 50, 1))  # 输入噪声
generated_sequence = generator.predict(noise)

# 将生成的数据反归一化回原始数据范围
generated_sequence = generated_sequence.reshape(-1, 1)  # 扁平化为二维数组
generated_sequence = scaler.inverse_transform(generated_sequence)
print("Generated Sequence:", generated_sequence)
