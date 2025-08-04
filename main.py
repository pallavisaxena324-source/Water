import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load training data
data = pd.read_csv('main_water.csv', encoding='latin-1')
print(data.head())
print(data.columns)

# Drop unnecessary columns
columns_to_drop = ['STATION CODE','LOCATIONS','STATE']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
print('Printing all data::')
print(data.head())
print(data.columns)

# Define classification function
def classify_safety(row):
    try:
        if (
            row["Temp"] <= 25 and
            row["D.O. (mg/l)"] >= 6 and
            row["PH"] <= 8.5 and
            row["CONDUCTIVITY (µmhos/cm)"] <= 300 and
            row["B.O.D. (mg/l)"] <= 2
        ):
            return 'Safe'
        else:
            return 'Unsafe'
    except KeyError as e:
        print(f"Column not found: {e}")
        return 'Unknown'

# Apply classification function
data['Safe/Unsafe'] = data.apply(classify_safety, axis=1)
print(data.head())

# Map 'Safe' and 'Unsafe' to binary values
data['Safe/Unsafe'] = data['Safe/Unsafe'].map({'Safe': 1, 'Unsafe': 0})
print(data.head())

# Split features and target
X = data.drop('Safe/Unsafe', axis=1, errors='ignore')
y = data['Safe/Unsafe']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


# Load a separate test dataset
test_data = pd.read_csv('test_data.csv', encoding='latin-1') # Replace with your test dataset path
print(test_data.head())

# Ensure test data has the same preprocessing
test_data = test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns], errors='ignore')

# Predict on test data
test_data['Prediction'] = knn_model.predict(test_data)

# Map binary predictions back to 'Safe' and 'Unsafe' (optional)
test_data['Prediction'] = test_data['Prediction'].map({1: 'Safe', 0: 'Unsafe'})

# Display predictions
print("Predictions on test data:")
print(test_data[['Prediction']])

