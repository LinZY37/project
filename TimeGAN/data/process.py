# import pandas as pd
#
# # 读取 stock_data.csv 文件
# file_path = 'stock_data.csv'
# stock_data = pd.read_csv(file_path)
#
# # 删除 'mud_level' 为 0 的行
# stock_data_cleaned = stock_data[stock_data['mud_level'] != 0]
#
# # 将清理后的数据保存到新的 CSV 文件
# cleaned_file_path = 'cleaned_stock_data.csv'
# stock_data_cleaned.to_csv(cleaned_file_path, index=False)
#
# # 打印清理后的数据的前几行
# print(stock_data_cleaned.head())
#



# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取数据
# df = pd.read_csv('stock_data.csv')
#
# # 设置 'ID' 为索引
# df['ID'] = range(1, len(df) + 1)
# df.set_index('ID', inplace=True)
#
# # 选择前 2000 条数据
# df_2000 = df.head(2000)
#
# # 创建一个画布
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # 绘制 mud_level 特征
# ax1.plot(df_2000.index, df_2000['mud_level'], label='Mud Level', color='blue')
# ax1.set_xlabel('ID')
# ax1.set_ylabel('Mud Level', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# # 创建第二个 y 轴，共享相同的 x 轴
# ax2 = ax1.twinx()
#
# # 绘制 rainfall_intensity 特征
# ax2.plot(df_2000.index, df_2000['rainfall_intensity'], label='Rainfall Intensity', color='green')
# ax2.set_ylabel('Rainfall Intensity', color='green')
# ax2.tick_params(axis='y', labelcolor='green')
#
# # 创建第三个 y 轴，共享相同的 x 轴
# ax3 = ax1.twinx()
#
# # 通过 offset，使第三个 y 轴不与第二个 y 轴重合
# ax3.spines['right'].set_position(('outward', 60))
#
# # 绘制 cumulative_rainfall 特征
# ax3.plot(df_2000.index, df_2000['cumulative_rainfall'], label='Cumulative Rainfall', color='red')
# ax3.set_ylabel('Cumulative Rainfall', color='red')
# ax3.tick_params(axis='y', labelcolor='red')
#
# # 添加标题
# plt.title('Mud Level, Rainfall Intensity, and Cumulative Rainfall Trend (First 2000 Rows)')
#
# # 显示图例
# fig.tight_layout()  # 自动调整布局，避免标签重叠
# plt.show()


# import pandas as pd
#
# # 读取数据
# df = pd.read_csv('generated_data.csv')
#
# # 保留所有数值列的五位小数
# df = df.round(5)
#
# # 将清理后的数据保存到新的 CSV 文件
# df.to_csv('rain_rounded.csv', index=False)
#
# # 打印前几行数据确认
# print(df.head())

#
# import pandas as pd
# import numpy as np
#
# # 读取数据
# df = pd.read_csv('test.csv')
#
# # 将 'time1' 和 'time2' 转换为 datetime 类型
# df['time1'] = pd.to_datetime(df['time1'])
# df['time2'] = pd.to_datetime(df['time2'])
#
# # 创建一个空的 'value' 列
# df['value'] = np.nan
#
# # 遍历 df 中每个 time1，找到对应时间差最小的 time2，并将 rain 值赋给 value
# for idx, row in df.iterrows():
#     # 获取当前的 time1
#     time1 = row['time1']
#
#     # 计算 time1 和所有 time2 的时间差
#     time_diff = abs(df['time2'] - time1)
#
#     # 找到时间差最小的索引
#     closest_idx = time_diff.idxmin()
#
#     # 将对应的 rain 值赋给 value 列
#     df.at[idx, 'value'] = df.at[closest_idx, 'rain']
#
# # 输出结果到新的 CSV 文件
# df.to_csv('output.csv', index=False)
#
# # 显示新文件路径
# print("The new file has been saved as 'output.csv'")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns







#
#
#
import pandas as pd

# # 读取数据
# data = pd.read_csv('generated_data.csv')

# # # 将数据中的所有数字乘以 0.65
# data = data * 0.75

# # 保留五位小数
# data = data.round(5)
#
# # 输出结果到新的 CSV 文件
# data.to_csv('mud0.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 读取数据
df = pd.read_csv('stock_data_2.csv')

# 设置 'ID' 为索引
df['ID'] = range(1, len(df) + 1)
df.set_index('ID', inplace=True)

# 选择前 2000 条数据
df_ = df.iloc[0:501]

# 绘制前 2000 条数据的 'mud_level' 走势
plt.figure(figsize=(30, 6))
plt.plot(df_.index, df_['mud_level'], label='Mud Level', color='blue')

# 添加标题和标签
plt.title('Mud Level Trend (First 2000 Rows)')
plt.xlabel('ID')
plt.ylabel('Mud Level')

# 关闭图例
plt.legend().set_visible(False)

# 显示图形
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# # 读取数据
# data = pd.read_csv('mud0.csv')
#
# # 将数据中的所有数字乘以 0.65
# data = data * 0.65
#
#
#
# data = data.round(4)
#
# mask = ((data >= 0.013) & (data <= 0.018))
# for column in data.columns:
#     # 获取该列中在 0.05 到 0.08 之间的数据
#     target_values = data[column][mask[column]]
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.3)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.025
#
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.15)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.0095
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.007
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.006
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.0065
#
#
# # 找到 0.05 到 0.08 之间的所有数据
# mask = ((data > 0.04) & (data < 0.09))
#
# # 随机抽取 80% 的这些数据，并赋值为 0.008
# for column in data.columns:
#     # 获取该列中在 0.05 到 0.08 之间的数据
#     target_values = data[column][mask[column]]
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.005)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.025
#
#     # # 随机选取 80% 的值并赋值为 0.008
#     # num_values_to_replace = int(len(target_values) * 0.05)
#     # indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # # 将选中的数据赋值为 0.008
#     # data.loc[indices_to_replace, column] = 0.008
#
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.45)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.0095
#
# mask = ((data >= 0.017) & (data <= 0.018))
# for column in data.columns:
#     # 获取该列中在 0.05 到 0.08 之间的数据
#     target_values = data[column][mask[column]]
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.105
#
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.0085
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.008
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.009
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.1)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.01
#
# # 输出结果到新的 CSV 文件
# data.to_csv('mud2.csv', index=False)
#
# def plot_feature_distribution(s, g, feature_name):
#     s[feature_name] = pd.to_numeric(s[feature_name], errors='coerce')  # Convert to numeric, coerce errors to NaN
#     g[feature_name] = pd.to_numeric(g[feature_name], errors='coerce')  # Convert to numeric, coerce errors to NaN
#
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(s[feature_name], label='real', linewidth=2)
#     sns.kdeplot(g[feature_name], label='generated', linestyle='--')
#     plt.title(f'{feature_name} Distribution Comparison')
#     plt.xlabel(feature_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# # 读取数据并转换为 DataFrame
# s = pd.read_csv("stock_data_mud.csv", header=None)
# g = pd.read_csv("mud2.csv", header=None)
#
# # 设置列名（根据实际情况修改列名）
# s.columns = ['mud_level']  # 假设数据只有一个特征，列名设置为 'mud_level'
# g.columns = ['mud_level']  # 同上
#
# # 示例：绘制 'mud_level' 的分布对比
# feature_name = 'mud_level'
# plot_feature_distribution(s, g, feature_name)
#
#
# from collections import Counter
#
# # 读取数据
# data = pd.read_csv('mud2.csv', header=None)
#
# # 确保将数据转换为浮动类型
# data = data.apply(pd.to_numeric, errors='coerce')
#
# # 将数据展平成一维数组
# all_data = data.values.flatten()
#
# # 使用 Counter 统计出现次数
# counter = Counter(all_data)
#
# # 获取出现次数最多的三个数字
# top_3_numbers = counter.most_common(20)
#
# # 输出结果
# print(top_3_numbers)




#
# # 读取数据
# data = pd.read_csv('mud0.csv')
# # 将数据中的所有数字乘以 0.65
# data = data * 0.65*0.8
#
# # 找到 0.05 到 0.08 之间的所有数据
# mask = ((data > 0.04) & (data < 0.08))
#
# # 随机抽取 80% 的这些数据，并赋值为 0.008
# for column in data.columns:
#     # 获取该列中在 0.05 到 0.08 之间的数据
#     target_values = data[column][mask[column]]
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.015)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.025
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.45)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = 0.0095
#
# mask = ((data >= 0.013) & (data <= 0.019))
# # 随机抽取 80% 的这些数据，并赋值为 0.008
# for column in data.columns:
#     # 获取该列中在 0.05 到 0.08 之间的数据
#     target_values = data[column][mask[column]]
#
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.4)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     # 强制转换为浮动类型
#     data.loc[indices_to_replace, column] = data.loc[indices_to_replace, column].astype(float) * 1.8
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.6)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = data.loc[indices_to_replace, column].astype(float) * 0.7
#
#     # 随机选取 80% 的值并赋值为 0.008
#     num_values_to_replace = int(len(target_values) * 0.2)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     data.loc[indices_to_replace, column] = data.loc[indices_to_replace, column].astype(float) * 0.5
#
# data = data.apply(pd.to_numeric, errors='coerce')
# data = data.astype(float)
# data = data.round(5)
# # 输出结果到新的 CSV 文件
# data.to_csv('mud3.csv', index=False)
#
#
#












#
#



# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取数据
# df = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/data/stock_data_.csv', header=None)
#
# # 设置 'ID' 为索引，假设数据只有一列，所以我们给这一列命名
# df['ID'] = range(1, len(df) + 1)
# df.set_index('ID', inplace=True)
#
# # 选择前 2000 条数据
# df_ = df.head(1000)
#
# # 绘制前 2000 条数据的趋势
# plt.figure(figsize=(30, 6))
# plt.plot(df_.index, df_.iloc[:, 0], label='Mud Level', color='blue')  # 这里使用 iloc 选择第一列数据
#
# # 添加标题和标签
# plt.title('Mud Level Trend (First 2000 Rows)')
# plt.xlabel('ID')
# plt.ylabel('Mud Level')
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.tight_layout()
# plt.show()







# import pandas as pd
#
# # 读取数据
# data = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/generated_data.csv')
#
# # 保留所有列的数据为五位小数
# data = data.round(5)
#
# # 将处理后的数据保存为新的CSV文件
# data.to_csv('/Users/Jerrylin/Downloads/TimeGAN-master/generated_data.csv', index=False)
#
# print("数据已处理并保留五位小数，已保存为 'processed_1.csv'")
#









# import pandas as pd
#
# # 读取1.csv文件
# df = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/generated_data1.csv')
#
# # 取第500到3000行的数据
# df_filtered = df.iloc[1:3001]
#
# # 输出到1.csv中
# df_filtered.to_csv('/Users/Jerrylin/Downloads/TimeGAN-master/data/stock_data.csv', index=False)
#


#
#
# import pandas as pd
#
# # 读取数据
# data = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/generated_data.csv')
#
# # 找到 0.05 到 0.08 之间的所有数据
# mask = ((data > 0.6))
# # 随机抽取 80% 的这些数据，并赋值为 0.008
# for column in data.columns:
#     # 获取该列中在 0.05 到 0.08 之间的数据
#     target_values = data[column][mask[column]]
#
#     num_values_to_replace = int(len(target_values) * 0.9)
#     indices_to_replace = np.random.choice(target_values.index, num_values_to_replace, replace=False)
#     # 将选中的数据赋值为 0.008
#     # data.loc[indices_to_replace, column] = 0.025
#     data.loc[indices_to_replace, column] = data.loc[indices_to_replace, column].astype(float) * 0.9
#
# # 保留五位小数
# data = data.round(5)
#
# # 输出结果到新的 CSV 文件
# data.to_csv('mud2.csv', index=False)
#
#

#
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def plot_feature_distribution_with_histogram(s, g, feature_name):
#     s[feature_name] = pd.to_numeric(s[feature_name], errors='coerce')  # Convert to numeric, coerce errors to NaN
#     g[feature_name] = pd.to_numeric(g[feature_name], errors='coerce')  # Convert to numeric, coerce errors to NaN
#
#     # Create a new figure
#     plt.figure(figsize=(12, 6))
#
#     # Plot KDE (Kernel Density Estimation) for both real and generated data
#     sns.kdeplot(s[feature_name], label='Real Data', linewidth=2)
#     sns.kdeplot(g[feature_name], label='Generated Data', linestyle='--', linewidth=2)
#
#     # Plot histograms for both real and generated data on top of KDE plots
#     plt.hist(s[feature_name], bins=30, alpha=0.5, label='Real Data (Histogram)', color='blue', density=True)
#     plt.hist(g[feature_name], bins=30, alpha=0.5, label='Generated Data (Histogram)', color='orange', density=True)
#
#     # Add titles and labels
#     plt.title(f'{feature_name} Distribution Comparison (KDE and Histogram)')
#     plt.xlabel(feature_name)
#     plt.ylabel('Density')
#
#     # Add legend
#     plt.legend()
#
#     # Display grid
#     plt.grid(True)
#
#     # Show the plot
#     plt.show()
#
# # Read data and convert to DataFrame
# g = pd.read_csv("/Users/Jerrylin/Downloads/TimeGAN-master/generated_data1.csv", header=None)
# s = pd.read_csv("/Users/Jerrylin/Downloads/TimeGAN-master/data/stock_data_2.csv", header=None).iloc[1001:2500]
#
# # Set column names (adjust according to your actual data)
# s.columns = ['mud_level']  # Assuming the data has only one feature, name it 'mud_level'
# g.columns = ['mud_level']  # Same as above
#
# # Plot for stock_data (real data)
# feature_name = 'mud_level'
# plot_feature_distribution_with_histogram(s, g, feature_name)
#
#


# # Create separate plots for each dataset (stock_data and mud2)
# # stock_data
# plt.figure(figsize=(12, 6))
# sns.kdeplot(s[feature_name], label='Real Data', linewidth=2)
# plt.hist(s[feature_name], bins=30, alpha=0.5, label='Real Data (Histogram)', color='blue', density=True)
# plt.title(f'{feature_name} Distribution for Stock Data')
# plt.xlabel(feature_name)
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # mud2 dataset
# plt.figure(figsize=(12, 6))
# sns.kdeplot(g[feature_name], label='Generated Data', linestyle='--', linewidth=2)
# plt.hist(g[feature_name], bins=30, alpha=0.5, label='Generated Data (Histogram)', color='orange', density=True)
# plt.title(f'{feature_name} Distribution for Generated Data (mud2)')
# plt.xlabel(feature_name)
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 假设 stock_data 是你的时间序列数据
# 例如，读取 stock_data.csv 文件
# stock_data = pd.read_csv('stock_data.csv')

#
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#
# # 读取原始数据和生成数据
# data_original = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/data/stock_data_.csv')
# data_generated = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/data/stock_data_2.csv')
#
# # 假设这两个数据文件中有 mud_level 列
# data_original = data_original['mud_level']
# data_generated = data_generated['mud_level']
#
# # 创建图形并设置大小
# plt.figure(figsize=(12, 6))
#
# # 绘制原始数据的自相关图（ACF）
# plt.subplot(1, 2, 1)  # 1行2列，第1个子图
# plot_acf(data_original, lags=40, title="Original Data ACF", ax=plt.gca())
#
# # 绘制生成数据的自相关图（ACF）
# plt.subplot(1, 2, 2)  # 1行2列，第2个子图
# plot_acf(data_generated, lags=40, title="Generated Data ACF", ax=plt.gca())
#
# plt.tight_layout()  # 调整子图间距
# plt.show()
#
# # 创建图形并设置大小
# plt.figure(figsize=(12, 6))
#
# # 绘制原始数据的偏自相关图（PACF）
# plt.subplot(1, 2, 1)  # 1行2列，第1个子图
# plot_pacf(data_original, lags=40, title="Original Data PACF", ax=plt.gca())
#
# # 绘制生成数据的偏自相关图（PACF）
# plt.subplot(1, 2, 2)  # 1行2列，第2个子图
# plot_pacf(data_generated, lags=40, title="Generated Data PACF", ax=plt.gca())
#
# plt.tight_layout()  # 调整子图间距
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('/Users/Jerrylin/Downloads/TimeGAN-master/joint_losses.csv')

# 创建图形
plt.figure(figsize=(10, 6))


plt.plot(data['Iteration'], data['Discriminator Loss'], label='Discriminator Loss', color='b')
plt.plot(data['Iteration'], data['Generator Loss (U)'], label='Generator Loss (U)', color='g')
plt.plot(data['Iteration'], data['Generator Loss (S)'], label='Generator Loss (S)', color='r')
plt.plot(data['Iteration'], data['Generator Loss (V)'], label='Generator Loss (V)', color='c')
plt.plot(data['Iteration'], data['Embedding Loss (T0)'], label='Embedding Loss (T0)', color='m')



# plt.plot(data['Iteration'], data['Supervised Loss'], label='Supervised Loss', color='b')

# 设置标题和标签
plt.title('Supervised Losses Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# 显示图例
plt.legend()


# # 去掉横坐标
# plt.xticks([])  # Remove x-axis ticks

# 去掉网格
plt.grid(False)


# 显示图形
plt.show()
