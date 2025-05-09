# project
Mudslide Warning System

This is a web-based system to monitor and forecast mudslide risks based on historical mud levels and rainfall data. It provides real-time data collection, predictive analytics, and weather monitoring to help mitigate mudslide disasters.

Features:
- Mudslide Level Forecasting: Predict mudslide risks based on the latest data.
- Real-Time Data Input: Users can submit real-time data on mud levels and rainfall.
- Weather Data Integration: Displays current weather, temperature, humidity, and wind power.
- Visualization: Displays graphs for rainfall and mud levels.
- Alert System: Displays warnings based on mudslide risk prediction.
- SQLite Database: Stores data related to sites, mud levels, rainfall, and predictions.

Requirements:
- Python 3.x
- Flask
- SQLAlchemy
- MinMaxScaler (for data scaling)

Installation

1. Download the warning system files

2. Install the dependencies:
   Make sure you have `pip` installed. You can install the required packages with:
   pip install -r requirements.txt

3. Database Setup:
   The application uses an SQLite database (`project.db`) to store site, mud level, and rainfall data. The database is automatically created upon the first run of the app.


Running the Application

1. Start the Flask server:
   python app.py
   The app will be available at http://127.0.0.1:5000/.

2. Access the web interface:
   Open your browser and go to http://127.0.0.1:5000/ to access the Mudslide Warning System interface.

Key Routes:
- / (index): Displays the main dashboard with data inputs, weather information, and mudslide risk alerts.
- /predict (GET): Sends a GET request to predict mud levels based on the last 12 hours of data.
- /add_data (POST): Adds new data (time, mud level, site) to the database.

Usage
1. Add Data:
   - Click on Add Data to input new data about mud levels and rainfall.
   - The data will be stored in the system and used for future predictions.

2. Mudslide Forecast:
   - The system uses historical data to predict the mudslide risk for the next hour.
   - The forecast value is displayed in the pie chart and updated based on the data entered.

3. Weather Information:
   - The system fetches real-time weather data for a specified city using AMap's API.
   - Weather info is displayed on the dashboard (e.g., temperature, humidity, wind power).

Data Structure

Mud Water Table (MudWater)
- id: Primary key.
- time: Timestamp of the recorded mud level.
- mud_level: The level of mud recorded at the given time.
- site_id: Foreign key referencing the site where the data was recorded.

Rainfall Table (Rainfall)
- id: Primary key.
- time: Timestamp of the recorded rainfall.
- rainfall: The amount of rainfall recorded at the given time.
- site_id: Foreign key referencing the site where the data was recorded.

Prediction Table (Prediction)
- id: Primary key.
- prediction_date: The date the mudslide risk prediction was made.
- prediction_result: The predicted mud level.
- mudwater_id: Foreign key referencing the corresponding mud water record.

Site Table (Site)
- id: Primary key.
- site: Name of the site.
- site_code: A unique code for the site.

How to Add New Data

1. Click on Add Data to open the modal.
2. Enter the Date and Time, Mud Level, and Site.
3. Click Upload Data to add it to the system.

Mudslide Prediction
- Once sufficient data is collected, the system will use a trained machine learning model to predict the mudslide risk for the next hour based on recent trends.

Contributing

Feel free to fork this repository, submit issues, or send pull requests. If you have any suggestions for improvements or bug fixes, please open an issue!

