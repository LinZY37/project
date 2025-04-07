import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import deque

# 加载数据集
df = pd.read_excel('data/initial_data.xlsx')
data = df[['降雨强度', '累积雨量', 'mud_level']].copy()
# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32)


# 设置序列长度
seq_length = 10
dataset = TimeSeriesDataset(scaled_data, seq_length=seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# TS-GAN生成器模型
class TS_Generator(nn.Module):
    def __init__(self, z_dim=100, output_dim=3, hidden_dim=128, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(z_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, z):
        # 扩展噪声为序列维度 [batch, seq, z_dim]
        z_seq = z.unsqueeze(1).repeat(1, self.seq_length, 1)

        # 生成完整序列
        lstm_out, _ = self.lstm(z_seq)  # [batch, seq, hidden]
        outputs = self.fc(lstm_out)  # [batch, seq, 3]

        return outputs


# TS-GAN判别器模型
class TS_Discriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]
        validity = self.fc(lstm_out[:, -1])  # 取最后时间步
        return torch.sigmoid(validity)


# 改进的损失函数
def feature_correlation_loss(fake_sequences):
    """计算序列级别的特征相关性"""
    cumulative_rain = fake_sequences[:, :, 1]  # 累积雨量
    mud_level = fake_sequences[:, :, 2]  # 泥石流等级

    # 计算时间维度上的协方差
    cov_matrix = torch.stack([
        torch.cov(torch.stack([cumulative_rain[:, t], mud_level[:, t]]))
        for t in range(fake_sequences.size(1))
    ])

    # 取平均协方差
    avg_cov = torch.mean(cov_matrix[:, 0, 1])
    return -torch.log(torch.sigmoid(avg_cov))


def temporal_consistency_loss(fake_sequences):
    """时序一致性损失"""
    first_diff = fake_sequences[:, 1:] - fake_sequences[:, :-1]
    second_diff = first_diff[:, 1:] - first_diff[:, :-1]
    return torch.mean(torch.abs(second_diff))


# 初始化模型
generator = TS_Generator(seq_length=seq_length)
discriminator = TS_Discriminator()
criterion = nn.BCELoss()

# 优化器设置
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 训练监控
loss_d_list = []
loss_g_list = []
gen_samples_queue = deque(maxlen=1000)  # 用于计算判别器准确率

# 训练循环
epochs = 40
for epoch in range(epochs):
    epoch_loss_d = 0.0
    epoch_loss_g = 0.0
    num_batches = 0

    for real_sequences in dataloader:
        batch_size = real_sequences.size(0)

        # ================= 训练判别器 =================
        optimizer_d.zero_grad()

        # 真实数据
        real_labels = torch.ones(batch_size, 1)
        real_pred = discriminator(real_sequences)
        loss_real = criterion(real_pred, real_labels)

        # 生成数据
        z = torch.randn(batch_size, 100)
        fake_sequences = generator(z)
        fake_labels = torch.zeros(batch_size, 1)
        fake_pred = discriminator(fake_sequences.detach())
        loss_fake = criterion(fake_pred, fake_labels)

        # 判别器总损失
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # ================= 训练生成器 =================
        optimizer_g.zero_grad()

        # 对抗损失
        gen_pred = discriminator(fake_sequences)
        loss_g_adv = criterion(gen_pred, real_labels)

        # 特征相关性损失
        loss_g_corr = feature_correlation_loss(fake_sequences)

        # 时序一致性损失
        loss_g_temp = temporal_consistency_loss(fake_sequences)

        # 总损失
        loss_g = loss_g_adv + 0.7 * loss_g_corr + 0.3 * loss_g_temp
        loss_g.backward()
        optimizer_g.step()

        # 记录数据
        epoch_loss_d += loss_d.item()
        epoch_loss_g += loss_g.item()
        num_batches += 1

        # 保存生成的样本
        gen_samples_queue.append(fake_sequences.detach().cpu())

    # ================= 监控与输出 =================
    avg_loss_d = epoch_loss_d / num_batches
    avg_loss_g = epoch_loss_g / num_batches
    loss_d_list.append(avg_loss_d)
    loss_g_list.append(avg_loss_g)
    print(f" G损失：{avg_loss_g:.4f} | D损失：{avg_loss_d:.4f}")

# 保存最终模型
torch.save(generator.state_dict(), 'timeseries_generator.pth')

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(loss_g_list, label='Generator Loss')
plt.plot(loss_d_list, label='Discriminator Loss')
plt.title("训练损失曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()


# 生成最终数据
def generate_sequences(num_sequences):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_sequences, 100)
        sequences = generator(z)
        data = sequences.view(-1, 3).numpy()
        return scaler.inverse_transform(data)


# 生成100个序列（1000个时间步）
final_data = generate_sequences(100)

# 创建DataFrame
generated_df = pd.DataFrame(final_data,
                            columns=['降雨强度', '累积雨量', 'mud_level'])

# 添加元数据
generated_df['sequence_id'] = np.repeat(np.arange(100), seq_length)
generated_df['time_step'] = np.tile(np.arange(seq_length), 100)

# 添加时间戳（假设起始时间与原始数据相同）
start_time = pd.to_datetime(df['time'].iloc[0])
time_stamps = pd.date_range(start=start_time,
                            periods=len(generated_df),
                            freq='10min')
generated_df['time'] = time_stamps

# 计算报警逻辑（同时满足两个条件）
generated_df['alert'] = 0
alert_mask = (generated_df['降雨强度'] > 8) & (generated_df['mud_level'] > 0.08)
generated_df.loc[alert_mask, 'alert'] = 1

# 保存数据
generated_df.to_excel('wgan.xlsx', index=False)

print("数据生成完成！前5个时间步示例：")
print(generated_df.head(10))
