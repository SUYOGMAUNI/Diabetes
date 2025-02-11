import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm import RandomForest  # Import your custom algorithm

# Load the dataset
df = pd.read_csv("updated_diabetes.csv")

# Prepare the data
y = df["Outcome"].values
X = df.drop("Outcome", axis=1).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the custom Random Forest model
model = RandomForest(n_trees=50, max_depth=10)

# Train the model
model.fit(X_train, y_train)

# Save the trained model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Test a sample prediction
test_features = np.array([[2, 148, 72, 25, 122.4, 27.8, 0.57, 34]])
test_features_scaled = scaler.transform(test_features)
prediction = model.predict(test_features_scaled)

# Output result
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

# Calculate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.4f}%")
