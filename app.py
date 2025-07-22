import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the trained model (which is a Pipeline) ---
try:
    model_pipeline = joblib.load("best_model.pkl")
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please ensure the trained model file is in the same directory as your Streamlit app.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best_model.pkl' is a valid joblib file.")
    st.stop()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# --- 2. Define Expected Features and Categories (CRUCIAL STEP) ---
# ************************************************************************************************
# IMPORTANT: YOU MUST REPLACE THE CONTENTS OF THESE LISTS WITH THE EXACT OUTPUT
#            FROM YOUR `train_models_final.py` SCRIPT AFTER RUNNING IT!
#            Copy-paste directly from your training script's terminal output.
# ************************************************************************************************

# Copy the output from "--- EXACT FEATURE NAMES USED FOR TRAINING (COPY THIS INTO STREAMLIT APP) ---" here.
# NOTE: 'fnlwgt' has been removed from this list as per your request.
EXPECTED_FEATURE_COLUMNS = [
    'age',
    'workclass',
    # 'fnlwgt', # Removed as per user's request
    'educational-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
]

# Copy the output from "--- CATEGORY ORDERS FOR LABEL ENCODED FEATURES (COPY INTO STREAMLIT APP) ---" here.
# Example (THESE ARE PLACEHOLDERS, REPLACE WITH YOUR ACTUAL LISTS):
WORKCLASS_CATEGORIES = ['Federal-gov', 'Local-gov', 'Others', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov']
MARITAL_STATUS_CATEGORIES = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
OCCUPATION_CATEGORIES = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Others', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
RELATIONSHIP_CATEGORIES = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
RACE_CATEGORIES = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
GENDER_CATEGORIES = ['Female', 'Male']
NATIVE_COUNTRY_CATEGORIES = ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']


# --- 3. Sidebar Inputs for Individual Prediction ---
st.sidebar.header("Input Employee Details")

# Numerical Inputs
age = st.sidebar.slider("Age", 17, 75, 30) # Adjusted range based on your cleaning
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Categorical Inputs (using selectboxes with predefined categories)
workclass = st.sidebar.selectbox("Work Class", WORKCLASS_CATEGORIES)
marital_status = st.sidebar.selectbox("Marital Status", MARITAL_STATUS_CATEGORIES)
occupation = st.sidebar.selectbox("Job Role", OCCUPATION_CATEGORIES)
relationship = st.sidebar.selectbox("Relationship", RELATIONSHIP_CATEGORIES)
race = st.sidebar.selectbox("Race", RACE_CATEGORIES)
gender = st.sidebar.selectbox("Gender", GENDER_CATEGORIES)
native_country = st.sidebar.selectbox("Native Country", NATIVE_COUNTRY_CATEGORIES)

# Note: 'educational-num', 'capital-gain', 'capital-loss' are
# not collected from the user directly but are expected by the model.
# They will be defaulted to 0 in the preprocessing function.

# --- 4. Preprocessing Function for Single Input ---
def preprocess_single_input(age, workclass, marital_status, occupation, relationship, race, gender, hours_per_week, native_country):
    # Create a dictionary to hold all feature values, initialized to 0 for numerical defaults
    data_dict = {col: [0] for col in EXPECTED_FEATURE_COLUMNS}

    # Populate numerical features directly from user input
    data_dict['age'] = [age]
    data_dict['hours-per-week'] = [hours_per_week]

    # Apply manual Label Encoding for categorical features
    # Use .index() for mapping, with a fallback for unseen categories (-1).
    data_dict['workclass'] = [WORKCLASS_CATEGORIES.index(workclass) if workclass in WORKCLASS_CATEGORIES else -1]
    data_dict['marital-status'] = [MARITAL_STATUS_CATEGORIES.index(marital_status) if marital_status in MARITAL_STATUS_CATEGORIES else -1]
    data_dict['occupation'] = [OCCUPATION_CATEGORIES.index(occupation) if occupation in OCCUPATION_CATEGORIES else -1]
    data_dict['relationship'] = [RELATIONSHIP_CATEGORIES.index(relationship) if relationship in RELATIONSHIP_CATEGORIES else -1]
    data_dict['race'] = [RACE_CATEGORIES.index(race) if race in RACE_CATEGORIES else -1]
    data_dict['gender'] = [GENDER_CATEGORIES.index(gender) if gender in GENDER_CATEGORIES else -1]
    data_dict['native-country'] = [NATIVE_COUNTRY_CATEGORIES.index(native_country) if native_country in NATIVE_COUNTRY_CATEGORIES else -1]

    # Create DataFrame from dictionary, explicitly setting column order
    processed_df = pd.DataFrame(data_dict, columns=EXPECTED_FEATURE_COLUMNS)

    # Ensure all columns are of integer type where expected (LabelEncoded outputs are int)
    # This loop ensures consistency, especially for the LabelEncoded columns and defaults.
    for col in EXPECTED_FEATURE_COLUMNS:
        if col in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', # 'fnlwgt' removed here
                    'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
            processed_df[col] = processed_df[col].astype(int)

    return processed_df # This DataFrame is already in the correct order and has all columns.

# Display Input Data
display_input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})
st.write("### 🔎 Input Data")
st.write(display_input_df)

# --- 5. Predict Button for Single Prediction ---
if st.button("Predict Salary Class"):
    try:
        processed_single_input = preprocess_single_input(age, workclass, marital_status, occupation, relationship, race, gender, hours_per_week, native_country)
        st.write("### Processed Input for Model:")
        st.write(processed_single_input) # Show what's being fed to the model

        # The loaded model_pipeline handles StandardScaler automatically
        prediction = model_pipeline.predict(processed_single_input)

        # Assuming your target encoder mapped '<=50K' to 0 and '>50K' to 1
        prediction_label = ">50K" if prediction[0] == 1 else "<=50K"
        st.success(f"✅ Prediction: Employee earns {prediction_label}")
    except Exception as e:
        st.error(f"Error during single prediction: {e}. This usually means the `EXPECTED_FEATURE_COLUMNS` or category mappings are still not an exact match to your trained model. Please re-verify the copied lists.")
        st.exception(e) # Display full traceback for debugging

# --- 6. Batch Prediction Section ---
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
# Changed to accept Excel files
uploaded_file = st.file_uploader("Upload an Excel file (.xlsx) for batch prediction", type="xlsx")

if uploaded_file is not None:
    try:
        batch_data = pd.read_excel(uploaded_file) # Changed to pd.read_excel
        st.write("Uploaded data preview:", batch_data.head())

        # --- Preprocessing for Batch Prediction ---
        # Initialize a DataFrame for batch with all EXPECTED_FEATURE_COLUMNS and default numerical values (0)
        processed_batch_df = pd.DataFrame(0, index=batch_data.index, columns=EXPECTED_FEATURE_COLUMNS)

        # Populate numerical features from batch_data (if present, else remain 0 as initialized)
        for col in ['age', 'hours-per-week', 'educational-num', 'capital-gain', 'capital-loss']: # 'fnlwgt' removed here
            if col in batch_data.columns:
                processed_batch_df[col] = batch_data[col]
            # If not in batch_data, it remains 0 as initialized

        # Apply manual Label Encoding for categorical features in batch data
        categorical_cols_to_process = [
            'workclass', 'marital-status', 'occupation', 'relationship',
            'race', 'gender', 'native-country'
        ]

        for col in categorical_cols_to_process:
            if col in batch_data.columns:
                # Get the category list dynamically
                category_list = eval(f"{col.upper().replace('-', '_')}_CATEGORIES")
                # Map values from batch_data using the category list's index, default to -1 for unseen
                processed_batch_df[col] = batch_data[col].astype(str).apply(
                    lambda x: category_list.index(x) if x in category_list else -1
                ).astype(int)
            else:
                st.warning(f"Warning: Batch data missing categorical column '{col}'. Defaulting to -1.")
                processed_batch_df[col] = -1 # Default for missing label-encoded categorical columns

        # Ensure all columns are of integer type where expected
        for col in EXPECTED_FEATURE_COLUMNS:
            if col in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', # 'fnlwgt' removed here
                        'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
                processed_batch_df[col] = processed_batch_df[col].astype(int)

        st.write("### Processed Batch Data for Model:")
        st.write(processed_batch_df.head()) # Show what's being fed to the model

        # The loaded model_pipeline handles StandardScaler automatically
        batch_preds = model_pipeline.predict(processed_batch_df)

        # Decode predictions back to original labels
        # Assuming your target encoder mapped '<=50K' to 0 and '>50K' to 1
        decoded_batch_preds = [">50K" if p == 1 else "<=50K" for p in batch_preds]
        batch_data['PredictedClass'] = decoded_batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())

        # Provide CSV download for results
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name='predicted_classes.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Error processing uploaded Excel file for batch prediction: {e}. Please ensure its columns match your model's expected inputs.")
        st.exception(e) # Display full traceback for debugging
