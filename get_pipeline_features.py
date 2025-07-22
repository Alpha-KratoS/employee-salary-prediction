import joblib
import pandas as pd

try:
    # Load your trained pipeline model
    model_pipeline = joblib.load("best_model.pkl")

    # The first step in your pipeline is StandardScaler.
    # We need the feature names that StandardScaler was fitted on.
    # These are typically the columns of the X_train DataFrame used to fit the pipeline.
    # If the StandardScaler itself has a feature_names_in_ attribute, that's ideal.
    # Otherwise, we rely on the X.columns from the training script.

    # Option 1: If the StandardScaler directly stores feature_names_in_ (less common for StandardScaler itself)
    # This might not work for StandardScaler, but is good for other transformers.
    # if hasattr(model_pipeline.named_steps['scaler'], 'feature_names_in_'):
    #     expected_features = list(model_pipeline.named_steps['scaler'].feature_names_in_)
    #     print("--- Features from StandardScaler (if available) ---")
    #     for feature in expected_features:
    #         print(f"'{feature}',")
    # else:
    # Option 2 (Most Reliable for your setup): Re-run your training script and copy the output
    # The training script explicitly prints the X.columns that were fed into the pipeline.
    print("Please re-run your 'train_models_final.py' script.")
    print("Copy the exact output from the section:")
    print("--- EXACT FEATURE NAMES USED FOR TRAINING (COPY THIS INTO STREAMLIT APP) ---")
    print("This is the definitive list of columns that your pipeline expects.")

except FileNotFoundError:
    print("Error: 'best_model.pkl' not found. Make sure it's in the same directory.")
except Exception as e:
    print(f"An error occurred while inspecting the model: {e}")
    print("Ensure 'best_model.pkl' is a valid scikit-learn Pipeline saved with joblib.")