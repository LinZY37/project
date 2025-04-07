import pandas as pd
import numpy as np

# 读取数据文件
df = pd.read_excel('updated_data_zexi.xlsx')  # 假设文件路径为 'data_zexi.xlsx'

# 确保时间戳列是日期时间格式
df['time'] = pd.to_datetime(df['time'])

# 设置时间戳为索引
df.set_index('time', inplace=True)

# 逐条判断是否符合高概率条件：mud_level > 0.1, rain > 0.1
for idx, row in df.iterrows():
    if row['mud_level'] > 0.09 and row['降雨强度'] > 0.09:
        # 打印符合条件的时间戳及相关数据
        print(f"高概率发生泥石流的时间戳：{idx}, id: {row['id']}, 降雨强度: {row['降雨强度']}, 累积雨量: {row['累积雨量']}")



# import pandas as pd
#
# # 读取 Excel 文件
# file_path = 'updated_data_zexi.xlsx'
# df = pd.read_excel(file_path)
#
# # 添加 'alert' 列，所有值设置为 0
# df['高程'] = 3080
# df['相对高差'] = 1520
# df['土壤类型'] = 2
# df['纵坡降'] = 0.092
# df['流域面积'] = 81.83
#
# # 保存更新后的 DataFrame 到新的 Excel 文件
# df.to_excel('1data_zexi.xlsx', index=False)
