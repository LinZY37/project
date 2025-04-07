# import pandas as pd
#
# # 读取 Excel 文件
# df = pd.read_excel('pt.xlsx')
#
# # 遍历数据，处理降雨强度为0时的累积雨量
# for i in range(1, len(df)):
#     if df.loc[i, '降雨强度'] == 0:  # 如果降雨强度为0
#         if df.loc[i-1, 'mud_level'] <= 0.001:  # 如果上一条数据的累积雨量小于等于0.001
#             df.loc[i, 'mud_level'] = 0  # 直接赋值0
#         else:
#             df.loc[i, 'mud_level'] = df.loc[i-1, 'mud_level'] - 0.001  # 上一条数据的累积雨量减去0.001
#
# # 将处理后的数据保存到 Excel 文件
# output_file = 'n_pt.xlsx'
# df.to_excel(output_file, index=False)

# import pandas as pd
#
# # 假设数据已加载到 DataFrame 中
# df = pd.read_excel('processed_with_previous_windowed_avg_water_level.xlsx')
#
# # 计算泥水位变化速率
# # 变化速率 = (当前泥水位 - 上一时刻泥水位) / 10分钟
# df['泥水位变化速率'] = df['mud_level'].diff() / (10 / 60)  # 10分钟换算为小时
#
# # 输出处理后的数据到新的 Excel 文件
# output_file = '22.xlsx'
# df.to_excel(output_file, index=False)
#

# import pandas as pd
#
# # 假设数据已加载到 DataFrame 中
# df = pd.read_excel('22.xlsx')
#
# i = 13
# # 计算每个时间窗口的最大降雨强度并将上一个窗口的最大降雨强度赋给该窗口内的所有数据
# for i in range(12, len(df)):  # 从第13个时间点开始处理
#     previous_window_max_rain = df.loc[i-12:i-1, '降雨强度'].max()  # 计算上一个窗口的最大降雨强度
#     if i - 12 >= 0:
#         df.loc[i-12:i-1, '上一个时间窗口的最大降雨强度'] = previous_window_max_rain  # 将上一个窗口的最大降雨强度赋给当前窗口内的所有数据
#
# # 输出处理后的数据到新的 Excel 文件
# output_file = '111.xlsx'
# df.to_excel(output_file, index=False)
#
# print(f"处理后的数据已保存为 {output_file}")

# import pandas as pd
#
# # 读取文件
# df = pd.read_excel('22.xlsx')
#
# # 假设时间列的名字是 'time'，如果实际列名不同，请修改
# df['time'] = pd.to_datetime(df['time'], errors='coerce')
#
# # 赋值 month 和 season 列
# df['month'] = df['time'].dt.month
#
# # 根据 month 给每条数据赋值 season（1春，2夏，3秋，4冬）
# def get_season(month):
#     if month in [3, 4, 5]:
#         return 1  # 春
#     elif month in [6, 7, 8]:
#         return 2  # 夏
#     elif month in [9, 10, 11]:
#         return 3  # 秋
#     else:
#         return 4  # 冬
#
# df['season'] = df['month'].apply(get_season)
#
# # 保存或显示结果
# df.to_excel('updated_22.xlsx', index=False)  # 也可以选择保存为新的文件
# print(df.head())


# import pandas as pd
#
# # 读取文件
# df = pd.read_excel('updated_22.xlsx')
#
# # 假设降雨强度列的名字是 'rain_intensity'，如果实际列名不同，请修改
# # 处理数据
# df['max'] = None  # 新增一列 'max'，用于存储最大值
#
# # 从第13条数据开始，往前找12条数据的降雨强度
# for i in range(12, len(df)):
#     # 获取当前数据及之前12条数据的降雨强度
#     previous_rain_intensity = df.loc[i - 12:i - 1, '降雨强度']
#
#     # 找出最大值
#     max_rain_intensity = previous_rain_intensity.max()
#
#     # 将最大值赋给当前行的 'max' 列
#     df.loc[i, 'max'] = max_rain_intensity
#
# # 保存或显示结果
# df.to_excel('12.xlsx', index=False)  # 也可以选择保存为新的文件

#
# import pandas as pd
#
# # 读取文件
# df = pd.read_excel('12.xlsx')
#
# # 假设泥沙水平列的名字是 'mud_level'，如果实际列名不同，请修改
# df['avg'] = None  # 新增一列 'avg'，用于存储计算的平均值
#
# # 从第13条数据开始，往前找12条数据的mud_level并计算平均值
# for i in range(12, len(df)):
#     # 获取当前数据及之前12条数据的mud_level
#     previous_mud_level = df.loc[i - 12:i - 1, 'mud_level']
#
#     # 计算平均值
#     avg_mud_level = previous_mud_level.mean()
#
#     # 将平均值赋给当前行的 'avg' 列
#     df.loc[i, 'avg'] = avg_mud_level
#
# # 保存或显示结果
# df.to_excel('new.xlsx', index=False)  # 保存为新的文件
# print(df.head())
#
#


import pandas as pd

# 读取 test.xlsx 文件
file_path = 'test.xlsx'  # 请确保文件路径正确
test_data = pd.read_excel(file_path)

# 保留前4321条数据
filtered_data = test_data.head(4321)

# 将处理后的数据保存到新的文件 xin.xlsx
output_path = 'xin.xlsx'  # 新文件保存路径
filtered_data.to_excel(output_path, index=False)

print(f"The filtered data has been saved as '{output_path}'")
