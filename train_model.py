# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('iris.csv')  # Replace with your CSV file path

# Preprocess the data
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Encode the target variable (species) into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(model, 'iris_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model trained and saved successfully!")