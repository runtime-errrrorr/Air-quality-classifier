# Air Quality Prediction System

## Overview
This project is a web-based air quality prediction system that uses machine learning to classify air quality based on various environmental parameters. The system can collect real-time sensor data and provide predictions about air quality levels.

## Features
- Real-time environmental data collection through serial connection
- Multiple room monitoring capability
- Air quality prediction using Random Forest model
- Interactive web interface for data input and visualization
- Support for various air quality parameters:
  - Temperature
  - Humidity
  - PM2.5 and PM10
  - NO2, SO2, and CO levels
  - Proximity to industrial areas
  - Population density

## Technical Stack
- **Backend**: Python Flask
- **Frontend**: HTML, CSS, Bootstrap 5
- **Machine Learning**: Scikit-learn (Random Forest model)
- **Data Processing**: Pandas
- **Model Storage**: joblib

## Project Structure
air-quality-predictor/
├── main.py # Main Flask application
├── random_forest_model.joblib # Pre-trained ML model
├── serial_data_connector.py # Serial data interface
├── static/
│ ├── css/
│ │ ├── style.css
│ │ └── location_select.css
│ └── js/
│ └── location_select.js
└── templates/
├── location_select.html # Location selection page
├── index.html # Main input form
└── result.html # Prediction results page

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Ensure the `random_forest_model.joblib` file is present in the root directory
2. Configure serial port settings in `serial_data_connector.py` for using hardware sensors

## Usage

1. Start the Flask application:
```bash
python main.py
```

2. Access the web interface at `http://localhost:5000`
3. Select a room location
4. Input or view sensor data
5. Get air quality predictions

## Air Quality Classifications

The system classifies air quality into four categories:
- **Good**: Clean air with low pollution levels
- **Moderate**: Acceptable air quality with some pollutants present
- **Poor**: Noticeable pollution that may affect sensitive groups
- **Hazardous**: Highly polluted air posing serious health risks


