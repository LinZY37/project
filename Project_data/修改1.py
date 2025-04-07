import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据集
df = pd.read_excel('/Users/Jerrylin/Downloads/毕设中期/Project_data/data/initial_data.xlsx')
data = df[['降雨强度', '累积雨量', 'mud_level']].copy()

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


dataset = SimpleDataset(scaled_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class ConstrainedGenerator(nn.Module):
    def __init__(self, z_dim=100, output_dim=3):
        super(ConstrainedGenerator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))

        # 非原地操作：创建新张量 ----------------------------
        # 原始特征
        rain_intensity = x[:, 0].unsqueeze(1)  # 降雨强度

        # 累积雨量（非负）
        cumulative_rain = torch.abs(x[:, 1]).unsqueeze(1)

        # mud_level（与累积雨量正相关）
        mud_level = cumulative_rain * 0.3 + x[:, 2].unsqueeze(1) * 0.1

        # 合并特征
        new_x = torch.cat([rain_intensity, cumulative_rain, mud_level], dim=1)
        return new_x


class Discriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


# 自定义损失函数（不变）
def feature_correlation_loss(fake_data):
    cumulative_rain = fake_data[:, 1]
    mud_level = fake_data[:, 2]
    cov = torch.mean((cumulative_rain - torch.mean(cumulative_rain)) *
                     (mud_level - torch.mean(mud_level)))
    return -torch.log(torch.sigmoid(cov))


# 初始化模型
generator = ConstrainedGenerator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练循环
epochs = 200
for epoch in range(epochs):
    for real_data in dataloader:
        batch_size = real_data.size(0)
        real_data = real_data.view(batch_size, -1)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 训练判别器
        optimizer_d.zero_grad()
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, real_labels)

        z = torch.randn(batch_size, 100)
        fake_data = generator(z)  # 生成器返回新张量
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        output_fake = discriminator(fake_data)
        loss_g_adv = criterion(output_fake, real_labels)
        loss_g_corr = feature_correlation_loss(fake_data)
        loss_g = loss_g_adv + 0.5 * loss_g_corr
        loss_g.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch + 1}/{epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')

# 生成数据及后处理（与之前相同）
# 生成新数据
z = torch.randn(1000, 100)
generated_data = generator(z)  # 形状 (1000, 3)

# 反标准化
generated_rescaled = scaler.inverse_transform(generated_data.detach().numpy())

# 创建DataFrame
generated_df = pd.DataFrame(generated_rescaled, columns=['降雨强度', '累积雨量', 'mud_level'])

# 添加静态特征（示例，根据实际数据调整）
generated_df['id'] = df['id'].iloc[:1000].values  # 每个id对应一行
generated_df['time'] = pd.date_range(start=df['time'].iloc[0], periods=1000, freq='10min')

# 计算alert（根据mud_level阈值）
generated_df['alert'] = (generated_df['mud_level'] > 0.09).astype(int)

# 保存
generated_df.to_excel('g_data.xlsx', index=False)
