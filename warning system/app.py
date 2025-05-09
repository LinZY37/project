from datetime import datetime

import numpy as np
import requests
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
# 初始化 Flask 应用
app = Flask(__name__)

# 配置 SQLite3 数据库 URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'  # 使用 SQLite 数据库，site.db 是数据库文件
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 禁用 SQLAlchemy 的对象修改追踪功能，减少开销

# 初始化数据库
db = SQLAlchemy(app)


# 位置表
class Site(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 主键
    site = db.Column(db.String(120), nullable=False)  # 位置名称
    site_code = db.Column(db.String(50), unique=True, nullable=False)  # 位置代码

    def __repr__(self):
        return f"Site('{self.site}', '{self.site_code}')"


# 泥水表
class MudWater(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 主键
    time = db.Column(db.DateTime, nullable=False)  # 时间（精确到小时）
    mud_level = db.Column(db.Float, nullable=False)  # 泥水位（至少五位小数）

    site_id = db.Column(db.Integer, db.ForeignKey('site.id'), nullable=False)  # 外键：位置id
    site = db.relationship('Site', backref=db.backref('mudwater', lazy=True))  # 反向关系

    def __repr__(self):
        return f"MudWater('{self.time}', '{self.mud_level}')"


# 雨量表
class Rainfall(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 主键
    time = db.Column(db.DateTime, nullable=False)  # 时间（精确到小时）
    rainfall = db.Column(db.Float, nullable=False)  # 雨量（至少五位小数）

    site_id = db.Column(db.Integer, db.ForeignKey('site.id'), nullable=False)  # 外键：位置id
    site = db.relationship('Site', backref=db.backref('rainfall', lazy=True))  # 反向关系

    def __repr__(self):
        return f"Rainfall('{self.time}', '{self.rainfall}')"


# 预测表
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # 主键
    prediction_date = db.Column(db.DateTime, nullable=False)  # 预测日期
    prediction_result = db.Column(db.Float, nullable=False)  # 预测结果（预测泥水位）

    mudwater_id = db.Column(db.Integer, db.ForeignKey('mud_water.id'), nullable=False)  # 外键：泥水位id
    mudwater = db.relationship('MudWater', backref=db.backref('prediction', lazy=True))  # 反向关系

    def __repr__(self):
        return f"Prediction('{self.prediction_date}', '{self.prediction_result}')"

# 在应用启动时自动创建数据库表
with app.app_context():
    db.create_all()

# 路由示例
@app.route('/')
def index():
    # 从数据库获取所有位置数据
    sites = Site.query.all()
    mud1 = MudWater.query.all()
    rain1 = Rainfall.query.all()

    mud1.sort(key=lambda m: m.time, reverse=True)
    rain1.sort(key=lambda r: r.time, reverse=True)

    # 获取过去144条数据，如果数据不够144条，则默认全部数据
    latest_mud_data = mud1[:144]

    # 初始化最大 mud_level 和相应的记录
    max_mud_level = None

    # 遍历获取的最新数据，找出 mud_level 最大的记录
    for m in latest_mud_data:
        if m.mud_level is not None:  # 确保mud_level不为None
            if max_mud_level is None or m.mud_level > max_mud_level:
                max_mud_level = m.mud_level

    # 将最大值存入变量 max
    max = max_mud_level
    max_mud_level = f"{max:.2f}"
    # 输出最大值（你可以根据需要进一步处理）
    print(f"Maximum mud_level: {max_mud_level}")



    # 获取过去144条数据，如果数据不够144条，则默认全部数据
    latest_rain_data = rain1[:144]

    # 初始化最大 mud_level 和相应的记录
    max_rain_level = None

    # 遍历获取的最新数据，找出 mud_level 最大的记录
    for m in latest_rain_data:
        if m.rainfall is not None:  # 确保mud_level不为None
            if max_rain_level is None or m.rainfall > max_rain_level:
                max_rain_level = m.rainfall

    # 将最大值存入变量 max
    max1 = max_rain_level
    max1 = f"{max1:.2f}"
    # 输出最大值（你可以根据需要进一步处理）
    print(f"Maximum mud_level: {max1}")





    mud = []
    for m in mud1:
        if m.mud_level is not None and m.time is not None:
            date_str = m.time.strftime('%H')  # 转换时间为月-日格式
            mud.append([date_str, m.mud_level])

    date_rain = []
    for r in rain1:
        if r.rainfall is not None and r.time is not None:
            date_str1 = r.time.strftime('%m-%d')
            date_rain.append([date_str1, r.rainfall])

    # 从数据库获取所有预测数据
    predictions = Prediction.query.all()

    api_key = "443c470ff873f0c1708dd871194bdc0d"
    city_code = "513329"
    url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_code}&key={api_key}"

    # Send the GET request to the API
    response = requests.get(url)

    weather_translation = {
        "晴": "Clear",
        "多云": "Cloudy",
        "阴": "Overcast",
        "阵雨": "Showers",
        "雷阵雨": "Thunderstorms",
        "小雨": "Light rain",
        "中雨": "Moderate rain",
        "大雨": "Heavy rain",
        "暴雨": "Rainstorm",
        "大暴雨": "Heavy rainstorm",
        "特大暴雨": "Severe rainstorm",
        "小雪": "Light snow",
        "中雪": "Moderate snow",
        "大雪": "Heavy snow",
        "暴雪": "Snowstorm",
        "浮尘": "Dust",
        "扬沙": "Dust storm",
        "沙尘暴": "Sandstorm",
        "雾": "Fog",
        "霾": "Haze"
    }


    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Check if the 'lives' field is in the response data
        if data.get("lives"):
            # Extract weather data from the response
            weather_info = data["lives"][0]
            weather = weather_info.get("weather")  # 天气
            temperature = weather_info.get("temperature")  # 温度
            humidity = weather_info.get("humidity")  # 湿度
            windpower = weather_info.get("windpower")  # 风力
            reporttime = weather_info.get("reporttime")
            weather_ = weather_translation.get(weather,
                                                         weather)  # Default to the original if no translation is found

            # Print the extracted weather information
            print(f"Weather: {weather_}")
            print(f"Temperature: {temperature}°C")
            print(f"Humidity: {humidity}%")
            print(f"Windpower: {windpower}")
        else:
            print("No weather data available")
    else:
        print("Failed to retrieve weather data")

    return render_template('index.html', sites=sites, predictions=predictions,mud = mud, date_rain=date_rain,
                           weather_=weather_, temperature=temperature,humidity=humidity,windpower=windpower,
                           reporttime = reporttime, max_mud_level=max_mud_level, max1=max1)

@app.route('/predict', methods=['GET'])
def predict():

    mud2 = MudWater.query.order_by(desc(MudWater.time)).limit(12).all()
    mud2 = sorted(mud2, key=lambda x: x.time)
    forecast_data = []
    for u in mud2:
        if u.mud_level is not None and u.time is not None:
            forecast_data.append(u.mud_level)

    # 2. 将 forecast_data 转换为 NumPy 数组，并归一化
    forecast_data_array = np.array(forecast_data).reshape(-1, 1)  # 将数据转为二维数组

    scaler = MinMaxScaler(feature_range=(0, 1))
    # 假设 scaler 是训练时用的 MinMaxScaler
    scaler.fit(forecast_data_array)  # 在这里拟合数据
    forecast_data_scaled = scaler.transform(forecast_data_array)

    # 3. 创建滑动窗口数据集
    def create_dataset(data, window_size=12):
        X = []
        for i in range(len(data) - window_size + 1):  # 滑动窗口，确保数据有足够的时间步
            X.append(data[i:i + window_size, 0])
        return np.array(X)

    X_forecast = create_dataset(forecast_data_scaled, window_size=12)


    X_forecast = X_forecast.reshape(X_forecast.shape[0], 12, 1)  # 每个样本包含 12 个时间步，每个时间步 1 个特征


    model = load_model('mud_level_predictor_model.h5', custom_objects={'loss': losses.MeanSquaredError()})


    y_pred_forecast = model.predict(X_forecast)


    y_pred_forecast_actual = scaler.inverse_transform(y_pred_forecast)
    y_pred_forecast_actual = np.round(y_pred_forecast_actual, 3)

    # 输出预测结果
    print("Predicted Mud Level: ", y_pred_forecast_actual)

    return jsonify({'prediction': y_pred_forecast_actual.tolist()})


@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        # 获取请求中的数据
        data = request.get_json()

        # 获取数据
        time = data.get('time')
        mud_level = data.get('mud_level')
        site_id = data.get('site_id')

        # 检查数据是否存在
        if not time or not mud_level or not site_id:
            return jsonify({"error": "Missing data"}), 400

        # 将时间字符串转换为 datetime 对象
        # 日期格式需要和前端传递的 datetime-local 格式一致
        time_obj = datetime.strptime(time, '%Y-%m-%dT%H:%M')

        # 创建新的 MudWater 记录
        new_mudwater = MudWater(time=time_obj, mud_level=mud_level, site_id=site_id)

        # 将数据插入数据库
        db.session.add(new_mudwater)
        db.session.commit()

        return jsonify({"message": "Data added successfully!"}), 200
    except Exception as e:
        db.session.rollback()  # 如果出错，回滚事务
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
