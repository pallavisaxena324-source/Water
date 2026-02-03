import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

MODEL_FILE = 'knn_model.joblib'

def load_data():
    # Load training data
    data = pd.read_csv('main_water.csv', encoding='latin-1')
    
    # Drop unnecessary columns
    columns_to_drop = ['STATION CODE','LOCATIONS','STATE']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
    
    return data

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

def train_model():
    data = load_data()
    
    # Apply classification function
    data['Safe/Unsafe'] = data.apply(classify_safety, axis=1)
    
    # Map 'Safe' and 'Unsafe' to binary values
    data['Safe/Unsafe'] = data['Safe/Unsafe'].map({'Safe': 1, 'Unsafe': 0})
    
    # Split features and target
    X = data.drop('Safe/Unsafe', axis=1, errors='ignore')
    y = data['Safe/Unsafe']
    
    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(knn_model, MODEL_FILE)
    
    return knn_model

def get_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return train_model()

def predict_water_quality(input_data):
    model = get_model()
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return 'Safe' if prediction == 1 else 'Unsafe'

def run_standalone():
    # Load test data
    test_data = pd.read_csv('test_data.csv', encoding='latin-1')
    print("Test Data:")
    print(test_data)
    
    model = get_model()
    
    # Predict on test data
    test_data['Prediction'] = model.predict(test_data)
    test_data['Prediction'] = test_data['Prediction'].map({1: 'Safe', 0: 'Unsafe'})
    
    print("\nPredictions:")
    print(test_data[['Prediction']])

if __name__ == '__main__':
    run_standalone()