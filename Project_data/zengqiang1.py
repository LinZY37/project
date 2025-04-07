#大改动 主要针对G loss过高 来源：ds 一点没动

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, data, seq_length=144, zero_threshold=0.1):
        self.data = data
        self.seq_length = seq_length
        # 过滤全零序列
        self.zero_mask = (np.abs(data) < zero_threshold).all(axis=1)
        self.valid_indices = self._find_valid_indices()

    def _find_valid_indices(self):
        valid = []
        for i in range(len(self.data) - self.seq_length):
            segment = self.data[i:i + self.seq_length]
            if not (np.all(np.abs(segment) < 0.1, axis=1).any()):
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        seq = self.data[start:start + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32)

# 改进的生成器模型
class TimeSeriesGenerator(nn.Module):
    def __init__(self, z_dim=100, output_dim=3, hidden_dim=128, seq_length=144):
        super().__init__()
        self.seq_length = seq_length

        self.init_layers = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 2),  # 修改输出维度为hidden_dim * 2
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2)
        )

        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=hidden_dim * 2,
            num_layers=1,
            batch_first=True,
        )

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        batch_size = z.size(0)

        # 初始变换
        h = self.init_layers(z)  # [batch, hidden*2]
        h = h.unsqueeze(0).repeat(1, 1, 1)  # [num_layers=2, batch, hidden*2]

        # 生成序列输入
        input_seq = z.unsqueeze(1).repeat(1, self.seq_length, 1)  # [batch, seq, z_dim]

        # LSTM处理
        lstm_out, _ = self.lstm(input_seq, (h, torch.zeros_like(h)))

        # 输出变换
        outputs = self.output_layers(lstm_out)
        return outputs



# 改进的判别器模型
class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.fc(context)


# 物理约束损失函数
def physics_constraint_loss(fake_sequences):
    """改进的物理约束损失"""
    # 输入形状: [batch, seq, 3]
    rain_intensity = fake_sequences[..., 0]
    cumulative_rain = fake_sequences[..., 1]
    mud_level = fake_sequences[..., 2]

    # 约束1：累积雨量单调非减
    cum_diff = cumulative_rain[:, 1:] - cumulative_rain[:, :-1]
    mono_loss = F.relu(-cum_diff).mean()  # 惩罚负差异

    # 约束2：泥石流等级与累积雨量正相关
    batch_size, seq_len = cumulative_rain.shape
    flat_cum = cumulative_rain.view(-1)
    flat_mud = mud_level.view(-1)

    cov_matrix = torch.cov(torch.stack([flat_cum, flat_mud]))
    corr_loss = -cov_matrix[0, 1]  # 最大化协方差

    # 约束3：降雨强度非负
    rain_nonneg = F.relu(-rain_intensity).mean()

    return 0.4 * mono_loss + 0.4 * corr_loss + 0.2 * rain_nonneg


# 梯度惩罚函数
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 最终生成函数
def generate_sequences(num_sequences):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_sequences, 100)
        sequences = generator(z)
        data = sequences.reshape(-1, 3).cpu().numpy()
        return scaler.inverse_transform(data)


if __name__ == '__main__':
    # 设置序列长度
    seq_length = 144
    dataset = TimeSeriesDataset(scaled_data, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # 初始化模型
    generator = TimeSeriesGenerator(seq_length=seq_length)
    discriminator = TimeSeriesDiscriminator()

    # 优化器设置
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, 0.9))

    # 学习率调度器
    # 使用 CosineAnnealingLR 或 StepLR 替代 CyclicLR
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=10)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=10)

    # 训练循环
    epochs = 40 #100
    for epoch in range(epochs):
        for i, real_sequences in enumerate(dataloader):
            batch_size = real_sequences.size(0)

            # ================= 训练判别器 =================
            optimizer_d.zero_grad()

            # 真实数据
            real_pred = discriminator(real_sequences)
            loss_real = -torch.mean(real_pred)

            # 生成数据
            z = torch.randn(batch_size, 100)
            with torch.no_grad():
                fake_sequences = generator(z)
            fake_pred = discriminator(fake_sequences)
            loss_fake = torch.mean(fake_pred)

            # 梯度惩罚
            gp = compute_gradient_penalty(discriminator,
                                          real_sequences.data,
                                          fake_sequences.data)

            # 总损失
            loss_d = loss_real + loss_fake + 10.0 * gp
            loss_d.backward()
            optimizer_d.step()

            # ================= 训练生成器 =================
            if i % 2 == 0:  # 每2个batch训练一次生成器
                optimizer_g.zero_grad()

                # 生成数据
                z = torch.randn(batch_size, 100)
                gen_sequences = generator(z)

                # 对抗损失
                gen_pred = discriminator(gen_sequences)
                loss_g_adv = -torch.mean(gen_pred)

                # 物理约束损失
                loss_physics = physics_constraint_loss(gen_sequences)

                # 总损失
                loss_g = loss_g_adv + 1.5 * loss_physics
                loss_g.backward()
                optimizer_g.step()

            # 更新学习率
            scheduler_g.step()
            scheduler_d.step()

        # 监控与保存
        with torch.no_grad():
            # 生成示例序列
            sample_z = torch.randn(5, 100)
            samples = generator(sample_z).cpu().numpy()
            samples = scaler.inverse_transform(samples.reshape(-1, 3))

            # 保存检查点
            torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')

        print(f'Epoch {epoch} | D Loss: {loss_d.item():.3f} | G Loss: {loss_g.item():.3f}')
        print(f'Physics Loss: {loss_physics.item():.3f} | GP: {gp.item():.3f}')

    # 生成并保存数据
    final_data = generate_sequences(100)
    generated_df = pd.DataFrame(final_data,
                                columns=['降雨强度', '累积雨量', 'mud_level'])
    generated_df.to_excel('improved_generated_data.xlsx', index=False)


