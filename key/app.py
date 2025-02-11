from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')  # If you used scaling in your model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [
            float(request.form['Pregnancy']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DPF']),
            float(request.form['Age'])
        ]

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale the features (if you used scaling during training)
        scaled_features = scaler.transform(features_array)
        print("Scaled Output:", scaled_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        print("Raw prediction output:", prediction)

        # Get prediction result
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return render_template('result.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
