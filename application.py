from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

application = Flask(__name__)
app = application


model = pickle.load(open('artifacts/model.pkl', 'rb'))

# Initialize LabelEncoder (for categorical features like 'cut', 'color', 'clarity')
le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()

# Fit the label encoders (assuming you already know the values for each category)
le_cut.fit(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
le_color.fit(['D', 'E', 'F', 'G', 'H', 'I', 'J'])
le_clarity.fit(['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2', 'IF'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        carat = float(request.form['carat'])
        cut = le_cut.transform([request.form['cut']])[0]
        color = le_color.transform([request.form['color']])[0]
        clarity = le_clarity.transform([request.form['clarity']])[0]
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        # Prepare the input data for prediction
        input_data = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])

        # Predict using the trained model
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
