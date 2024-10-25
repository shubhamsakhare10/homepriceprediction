from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')

# Load the pipeline correctly
try:
    pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    pipe = None

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model pipeline is loaded
    if pipe is None:
        return "Model pipeline not loaded properly. Please check the model file."

    # Get form data
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    # Print the raw input data
    print("Raw Input Data:")
    print(input_data)

    # Convert fields to numeric, using errors='coerce' to handle non-numeric values
    input_data['beds'] = pd.to_numeric(input_data['beds'], errors='coerce')
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')
    input_data['size'] = pd.to_numeric(input_data['size'], errors='coerce')
    input_data['zip_code'] = pd.to_numeric(input_data['zip_code'], errors='coerce')

    # Print the data after numeric conversion
    print("Input Data After Conversion to Numeric:")
    print(input_data)

    # Handle missing values by filling with the median of each column from the training data
    input_data = input_data.fillna(data.median())

    # Print the data after handling missing values
    print("Input Data After Handling Missing Values:")
    print(input_data)

    # Ensure that the input data matches the format expected by the pipeline
    try:
        # Make a prediction using the model
        prediction = pipe.predict(input_data)[0]
        return f"Price: INR {prediction}"
    except AttributeError as e:
        # If an AttributeError occurs, check if it's related to transform
        print(f"Error during prediction: {e}")
        return "An error occurred during prediction. Please check your input values."
    except Exception as e:
        # Catch other exceptions
        print(f"Unexpected error during prediction: {e}")
        return "An error occurred during prediction. Please check your input values."

if __name__ == "__main__":
    app.run(debug=True, port=5000)
