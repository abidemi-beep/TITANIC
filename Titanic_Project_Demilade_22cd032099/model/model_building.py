"""
Titanic Survival Prediction Model Development
This script trains a Random Forest Classifier to predict Titanic passenger survival
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION MODEL DEVELOPMENT")
    print("=" * 60)

    # Step 1: Load the Titanic dataset
    print("\n[1] Loading Titanic Dataset...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    # Step 2: Feature Selection
    print("\n[2] Feature Selection...")
    selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    target = 'Survived'

    df_selected = df[selected_features + [target]].copy()
    print(f"Selected Features: {selected_features}")
    print(f"Target Variable: {target}")
    print(f"\nDataset shape after selection: {df_selected.shape}")

    # Step 3: Data Preprocessing
    print("\n[3] Data Preprocessing...")

    # 3a. Handling Missing Values
    print("\nMissing values before handling:")
    print(df_selected.isnull().sum())

    # Fill missing Age with median
    df_selected['Age'].fillna(df_selected['Age'].median(), inplace=True)

    # Fill missing Fare with median
    df_selected['Fare'].fillna(df_selected['Fare'].median(), inplace=True)

    # Fill missing Embarked with mode
    df_selected['Embarked'].fillna(df_selected['Embarked'].mode()[0], inplace=True)

    print("\nMissing values after handling:")
    print(df_selected.isnull().sum())

    # 3b. Encoding Categorical Variables
    print("\n[3b] Encoding Categorical Variables...")

    # Encode Sex: male=1, female=0
    le_sex = LabelEncoder()
    df_selected['Sex'] = le_sex.fit_transform(df_selected['Sex'])
    print(f"Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

    # Encode Embarked: C=0, Q=1, S=2
    le_embarked = LabelEncoder()
    df_selected['Embarked'] = le_embarked.fit_transform(df_selected['Embarked'])
    print(f"Embarked encoding: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

    print("\nData after preprocessing:")
    print(df_selected.head())
    print(f"\nData types:\n{df_selected.dtypes}")

    # Step 4: Split Data
    print("\n[4] Splitting Data into Training and Testing Sets...")
    X = df_selected[selected_features]
    y = df_selected[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Class distribution in training set:\n{y_train.value_counts()}")

    # Step 5: Feature Scaling
    print("\n[5] Feature Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed using StandardScaler")

    # Step 6: Model Training - Random Forest Classifier
    print("\n[6] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)
    print("Model training completed!")

    # Feature Importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)

    # Step 7: Model Evaluation
    print("\n[7] Model Evaluation...")
    y_pred = model.predict(X_test_scaled)

    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

    # Step 8: Save the Model and Preprocessing Objects
    print("\n[8] Saving Model and Preprocessing Objects...")

    # Create a dictionary with all necessary objects
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': selected_features,
        'sex_encoder': le_sex,
        'embarked_encoder': le_embarked
    }

    # Save to disk
    joblib.dump(model_package, 'titanic_survival_model.pkl')
    print("Model saved as 'titanic_survival_model.pkl'")

    # Step 9: Demonstrate Model Reloading and Prediction
    print("\n[9] Demonstrating Model Reloading and Prediction...")

    # Reload the model
    loaded_package = joblib.load('titanic_survival_model.pkl')
    loaded_model = loaded_package['model']
    loaded_scaler = loaded_package['scaler']

    print("Model reloaded successfully!")

    # Test with sample data
    print("\n--- Testing with Sample Passengers ---")

    sample_passengers = [
        {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 29,
            'Fare': 80.0,
            'Embarked': 'C'
        },
        {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 25,
            'Fare': 8.05,
            'Embarked': 'S'
        }
    ]

    for i, passenger in enumerate(sample_passengers, 1):
        print(f"\nPassenger {i}: {passenger}")
        
        # Prepare input
        passenger_encoded = passenger.copy()
        passenger_encoded['Sex'] = 1 if passenger['Sex'] == 'male' else 0
        embarked_map = {'C': 0, 'Q': 1, 'S': 2}
        passenger_encoded['Embarked'] = embarked_map[passenger['Embarked']]
        
        # Create DataFrame
        input_df = pd.DataFrame([passenger_encoded])[selected_features]
        
        # Scale and predict
        input_scaled = loaded_scaler.transform(input_df)
        prediction = loaded_model.predict(input_scaled)[0]
        probability = loaded_model.predict_proba(input_scaled)[0]
        
        result = "SURVIVED" if prediction == 1 else "DID NOT SURVIVE"
        print(f"Prediction: {result}")
        print(f"Survival Probability: {probability[1]:.2%}")

    print("\n" + "=" * 60)
    print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nModel file 'titanic_survival_model.pkl' has been created.")
    print("You can now use this file in your Flask application.")


if __name__ == "__main__":
    main()