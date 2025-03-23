from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "random_forest_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Define feature names (must match the training data)
FEATURE_NAMES = [
    'Temperature',
    'Humidity',
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO',
    'Proximity_to_Industrial_Areas',
    'Population_Density'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_info = None
    error = None

    if request.method == 'POST':
        try:
            # List of expected fields
            fields = [
                'temperature',
                'humidity',
                'pm25',
                'pm10',
                'no2',
                'so2',
                'co',
                'proximity_industrial',
                'population_density'
            ]
            
            # Retrieve and validate each field without using default values
            values = {}
            for field in fields:
                value = request.form.get(field)
                if value is None or value.strip() == "":
                    raise ValueError(f"{field.replace('_', ' ').capitalize()} is required.")
                try:
                    values[field] = float(value)
                except ValueError:
                    raise ValueError(f"{field.replace('_', ' ').capitalize()} must be a valid number.")

            # Validate that all numbers are non-negative
            if any(v < 0 for v in values.values()):
                raise ValueError("All inputs must be non-negative numbers.")

            # Create a DataFrame with the inputs using exact column names from the model
            input_data = pd.DataFrame({
                'Temperature': [values['temperature']],
                'Humidity': [values['humidity']],
                'PM2.5': [values['pm25']],
                'PM10': [values['pm10']],
                'NO2': [values['no2']],
                'SO2': [values['so2']],
                'CO': [values['co']],
                'Proximity_to_Industrial_Areas': [values['proximity_industrial']],
                'Population_Density': [values['population_density']]
            })

            # Ensure the column order matches the training data
            input_data = input_data[FEATURE_NAMES]

            # Predict air quality
            prediction = model.predict(input_data)[0]
            
            # Map prediction to detailed category
            air_quality_categories = {
                0: {
                    'level': 'Good',
                    'description': 'Clean air with low pollution levels.',
                    'class': 'alert-success'
                },
                1: {
                    'level': 'Poor',
                    'description': 'Noticeable pollution that may cause health issues for sensitive groups.',
                    'class': 'alert-danger'
                },
                2: {
                    'level': 'Hazardous',
                    'description': 'Highly polluted air posing serious health risks to the population.',
                    'class': 'alert-dark'
                },
                3: {
                    
                    'level': 'Moderate',
                    'description': 'Acceptable air quality but with some pollutants present.',
                    'class': 'alert-warning'
                }
            }
            
            prediction_info = air_quality_categories.get(prediction, {
                'level': 'Unknown',
                'description': 'Unable to determine air quality level.',
                'class': 'alert-secondary'
            })
            print(f"Prediction made: {prediction_info}")  # Debug print
        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"
    else:
        print("GET request - prediction_info should be None")  # Debug print

    print(f"Returning prediction_info: {prediction_info}")  # Debug print
    return render_template('index.html', prediction=prediction_info, error=error)

if __name__ == '__main__':
    app.run(debug=True)
