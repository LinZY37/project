import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据文件
real_data = pd.read_excel('data/initial_data.xlsx')
fake_data = pd.read_excel('generated_sequences.xlsx')

# 特征分布对比图
# def plot_feature_distribution(real_data, fake_data, feature_name):
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(real_data[feature_name], label='real', linewidth=2)
#     sns.kdeplot(fake_data[feature_name], label='generated', linestyle='--')
#     plt.title(f'{feature_name}VS')
#     plt.xlabel(feature_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# # 示例：绘制降雨强度（'降雨强度'为特征名）的分布对比
# feature_name = 'mud_level'  # 你可以根据实际需要替换其他特征名
# plot_feature_distribution(real_data, fake_data, feature_name)
#

# 5. 特征相关性矩阵
def plot_correlation_heatmap():
    # 计算相关系数矩阵
    real_corr = real_data[['降雨强度', '累积雨量', 'mud_level']].corr()
    fake_corr = fake_data[['降雨强度', '累积雨量', 'mud_level']].corr()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(real_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax[0])
    ax[0].set_title('Real Data Correlation')

    sns.heatmap(fake_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax[1])
    ax[1].set_title('Generated Data Correlation')

    plt.tight_layout()
    plt.show()


plot_correlation_heatmap()