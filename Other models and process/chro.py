import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline


print(torch.cuda.is_available())  #
pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-small",
  device_map="cpu",
  # torch_dtype=torch.bfloat16,
)

df = pd.read_csv("stock_data_2.csv")
real = df["mud_level"].iloc[501:516].values
df = df.head(500)



# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["mud_level"])
prediction_length = 15
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# visualize the forecast
forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


# 假设 median 是预测的中位数，real 是实际的 24 条数据
forecast_values = low  # 24 个预测值
real_values = real  # 真实的 24 条数据

# 计算相对误差
relative_error = np.abs(forecast_values - real_values) / np.abs(real_values)

# 计算平均相对误差 (Mean Relative Error)
mre = np.mean(relative_error)

# 打印结果，并避免科学计数法
# 设置打印的精度，避免使用科学计数法
np.set_printoptions(precision=8, suppress=True)

plt.figure(figsize=(30, 6))
plt.plot(df["mud_level"], color="royalblue", label="historical data")
plt.plot(forecast_index, high, color="red", label="median forecast")
plt.plot(forecast_index, real, color="green", linestyle='--', label="Actual Future Data")

plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
# 关闭图例
plt.legend().set_visible(False)

# 关闭网格线
plt.grid(False)

plt.show()
plt.savefig('forecast_plot.png')
