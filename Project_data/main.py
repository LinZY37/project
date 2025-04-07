




#多项式插值 加上累积雨量逻辑
# import pandas as pd
# import numpy as np
#
# # 读取 Excel 文件
# df = pd.read_excel('rain1.xlsx')
#
# # 确保时间戳列是日期时间格式
# df['time'] = pd.to_datetime(df['time'])
#
# # 设置时间戳为索引
# df.set_index('time', inplace=True)
#
# # 生成从 2024/4/1 00:00:00 到 2024/10/31 23:22:20，每隔 10 分钟的时间序列
# start_time = pd.to_datetime('2024-04-01 00:00:00')
# end_time = pd.to_datetime('2024-04-30 23:22:20')
# time_index = pd.date_range(start=start_time, end=end_time, freq='10T')
#
# # 将时间戳对齐到标准时间索引
# df_resampled = df.reindex(time_index)
#
# # 对雨量和降水强度进行多项式插值，阶数为 2
# df_resampled['雨量'] = df_resampled['雨量'].interpolate(method='polynomial', order=2)
# df_resampled['rain'] = df_resampled['rain'].interpolate(method='polynomial', order=2)
#
# # 处理负值，将负值替换为0
# df_resampled['雨量'] = np.maximum(df_resampled['雨量'], 0)
#
# # 处理接近零的值，将接近零的值设置为零（例如小于0.01的值）
# threshold = 0.001  # 根据实际情况调整阈值
# df_resampled['雨量'] = np.where(df_resampled['雨量'] < threshold, 0, df_resampled['雨量'])
#
# # 对降水强度做类似处理
# df_resampled['rain'] = np.maximum(df_resampled['rain'], 0)
# df_resampled['rain'] = np.where(df_resampled['rain'] < threshold, 0, df_resampled['rain'])
#
# # 计算累积雨量并修正 rain 值
# df_resampled['cumulative_rain'] = df_resampled['雨量'].cumsum()  # 计算累积雨量
#
# # 修正 'rain' 值：当前雨量加上前一个 rain 的值
# df_resampled['rain'] = df_resampled['cumulative_rain'].shift(1, fill_value=0)
#
# # 保存插值后的数据到新的 Excel 文件
# df_resampled.to_excel('rain1_interpolated_polynomial_processed_with_corrected_rain.xlsx')
#
# print("数据插值完成，已保存到 'rain1_interpolated_polynomial_processed_with_corrected_rain.xlsx'")



# import pandas as pd
# import numpy as np
#
# # 读取 Excel 文件
# df = pd.read_excel('rain1.xlsx')
#
# # 确保时间戳列是日期时间格式
# df['time'] = pd.to_datetime(df['time'])
#
# # 设置时间戳为索引
# df.set_index('time', inplace=True)
#
# # 生成从 2024/4/1 00:00:00 到 2024/10/31 23:22:20，每隔 10 秒的时间序列
# start_time = pd.to_datetime('2024-04-01 00:00:00')
# end_time = pd.to_datetime('2024-10-31 23:22:20')
# time_index = pd.date_range(start=start_time, end=end_time, freq='10T')
#
# # 将时间戳对齐到标准时间索引
# df_resampled = df.reindex(time_index)
#
# # 对雨量和降水强度进行线性插值
# df_resampled['累积雨量'] = df_resampled['累积雨量'].interpolate(method='linear')
# df_resampled['降雨强度'] = df_resampled['降雨强度'].interpolate(method='linear')
#
# # 保存插值后的数据到新的 Excel 文件
# df_resampled.to_excel('rain1_correct.xlsx')
#
# print("数据插值完成，已保存到 'rain1_interpolated.xlsx'")



#线性插值
# import pandas as pd
# import numpy as np
#
# # 读取 Excel 文件
# df = pd.read_excel('mud1.xlsx')
#
# # 确保时间戳列是日期时间格式
# df['time'] = pd.to_datetime(df['time'])
#
# # 设置时间戳为索引
# df.set_index('time', inplace=True)
#
# # 生成从 2024/4/1 00:00:00 到 2024/10/31 23:22:20，每隔 10 T的时间序列
# start_time = pd.to_datetime('2024-04-01 00:00:00')
# end_time = pd.to_datetime('2024-10-31 23:22:20')
# time_index = pd.date_range(start=start_time, end=end_time, freq='10T')
#
# # 将时间戳对齐到标准时间索引
# df_resampled = df.reindex(time_index)
#
# # 对雨量和降水强度进行线性插值
# df_resampled['mud_level'] = df_resampled['mud_level'].interpolate(method='linear')
#
# # 保存插值后的数据到新的 Excel 文件
# df_resampled.to_excel('mud1_correct.xlsx')
#
# print("数据插值完成，已保存到 'mud1_correct.xlsx'")



import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = '/Users/Jerrylin/Downloads/Project_data/data_zexi.xlsx'  # 替换为您的CSV文件路径
df = pd.read_excel(file_path, engine='openpyxl')# 假设数据列名为 '时间' 和 '泥水位'
df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M')

# 绘制时间与泥水位的关系图
plt.figure(figsize=(100, 6))
plt.plot(df['time'], df['累积雨量'], marker='o', linestyle='-', color='b', label='累积雨量')

# 添加标题和标签
plt.title('time vs 累积雨量')
plt.xlabel('time')
plt.ylabel('累积雨量')
plt.xticks(rotation=45)  # 旋转x轴标签以便显示
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.legend()
plt.show()


