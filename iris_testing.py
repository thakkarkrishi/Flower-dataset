import pandas as pd
import pickle

# Load Model
with open("D:/data science/flower dataset/iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Label Encoder
with open("D:/data science/flower dataset/iris_encoder.pkl", "rb") as file:
    le = pickle.load(file)

# New Flower Data
new_data = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})

# Prediction
prediction = model.predict(new_data)

# Convert Encoded Prediction
species = le.inverse_transform(prediction)

print("Predicted Species:", species[0])
