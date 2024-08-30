# color_match_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
df = pd.read_csv('Book1.csv')

# Check the first few rows of the dataset to understand its structure
print(df.head())

# Preprocessing
# Assuming 'Color' is the input feature and 'BestMatch' is the target variable
X = df[['Color']]
y = df['BestMatch']

# Encode categorical features
label_encoder_X = LabelEncoder()
X_encoded = label_encoder_X.fit_transform(X['Color']).reshape(-1, 1)  # Reshape to ensure it's a 2D array

label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model and encoders
joblib.dump(model, 'model/color_match_model.pkl')
joblib.dump(label_encoder_X, 'model/label_encoder_X.pkl')
joblib.dump(label_encoder_y, 'model/label_encoder_y.pkl')

print("Model and encoders saved successfully.")
