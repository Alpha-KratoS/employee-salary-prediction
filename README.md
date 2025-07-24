# ___**# Welcome to the employee-salary-prediction wiki!**___


## # Employee Salary Prediction using LightGBM Machine Learning and Interactive Data Insights# 
***


This project aims to develop a robust machine learning model to predict employee salaries based on various features, and to provide interactive data insights through a user-friendly web application. This project was developed as a capstone for the IBM & Edunet Foundation AI/ML Internship.
***
## # Project Structure
***
he project is organized into a modular and scalable structure:

employee_salary_prediction/ ├── data/ │ └── employee details.csv # The raw dataset used for training and insights. ├── src/ │ ├── init.py # Makes 'src' a Python package. │ ├── preprocess.py # Contains functions for data loading, cleaning, and preprocessing. │ ├── train.py # Handles the machine learning model training and evaluation. │ └── app.py # (Optional) Contains functions for making new predictions using the saved model. ├── models/ │ └── best_model.pkl # The saved trained LightGBM model pipeline. ├── notebooks/ │ └── analy_modelling.ipynb # Optional: Jupyter Notebook for exploratory data analysis and experimentation. ├── requirements.txt # Lists all Python dependencies required for the project. ├── README.md # This project documentation file. ├── analy_modelling.py # Main script to orchestrate the data preprocessing, model training, and saving. └── app.py # The Streamlit web application for interactive prediction and insights.
***
## Problem Statement
***
In today's dynamic job market, accurately determining employee salaries is crucial for both businesses and individuals. Companies need fair compensation strategies for effective budgeting, talent acquisition, and employee retention. Simultaneously, employees benefit from understanding the various factors that influence their earning potential and career progression. Traditional methods of salary determination can often be subjective, leading to inconsistencies and a lack of data-driven insights. This project addresses the challenge of predicting employee salaries by leveraging advanced machine learning techniques, aiming to provide data-driven estimations and interactive insights into salary trends.

Note: This project utilizes a simulated/synthetic dataset for demonstration purposes. Therefore, the predicted salary values and observed salary distributions are based on the patterns within this simulated data and may not reflect actual real-world salary trends or magnitudes.
***
## System Development Approach (Technology Used)
***
Overall Strategy: An end-to-end Machine Learning (ML) pipeline approach was adopted to ensure a structured, reproducible, and scalable workflow. This pipeline encompasses all stages from data acquisition and rigorous preprocessing to advanced model training, comprehensive evaluation, and interactive web application deployment.

System Requirements:

Operating System: Windows, macOS, or Linux
Software: Python 3.8+
Environment Management: Virtual environment (e.g., venv or conda) is highly recommended to manage project dependencies.
Development Environment: Visual Studio Code, Jupyter Notebooks, or any compatible Integrated Development Environment (IDE).
User Interface: A modern web browser is required to interact with the Streamlit application.

Libraries Required to Build the Model and Application:

pandas: Essential for efficient data loading, manipulation, and cleaning operations.
scikit-learn: Utilized for fundamental machine learning tasks including data splitting (train/test sets), robust preprocessing techniques (StandardScaler for numerical features, OneHotEncoder for categorical features), and constructing the overall ML Pipeline.
lightgbm: The core machine learning algorithm, specifically LGBMRegressor, chosen for its high performance, speed, and efficiency in handling large tabular datasets for regression tasks.
matplotlib & seaborn: Powerful libraries for creating static and statistical data visualizations, used extensively for model evaluation plots and generating data insights within the Streamlit application.
streamlit: The open-source Python framework used to rapidly build and deploy the interactive and user-friendly web application (front-end) for the project.
joblib: Employed for saving the trained machine learning model pipeline to disk and loading it back for predictions, ensuring model persistence across sessions.
numpy: Provides essential numerical computing capabilities, often used implicitly by other libraries like pandas and scikit-learn.
***
## Algorithm & Deployment (Step by Step Procedure)
***
The project follows a systematic procedure to ensure robust development and deployment:

Data Loading & Initial Inspection:

The project initiates by loading the employee_details.csv file into a pandas DataFrame.
An initial inspection is performed to understand the dataset's structure, data types, identify any missing values, and view basic descriptive statistics of the features.
Data Preprocessing & Cleaning:

Irrelevant Column Removal: Columns such as Employee_ID (a unique identifier) and Company_Name (highly granular, potentially noisy for general prediction) were dropped.
Robust Categorical Data Cleaning:
Standardization: Inconsistent entries within categorical features (e.g., "full time", "full-time", "Full Time" in Employment_Type were consolidated to a single "full-time" representation).
Typo Correction: Obvious typos and similar categories (e.g., "contracontractor" and "contract" were mapped to "contractor") were standardized.
Formatting Consistency: Leading/trailing spaces were removed, and consistent casing (lowercase for model training, Title Case for UI display) was applied across all relevant categorical features (Job_Role, Location, Gender, Education_Level, etc.).
Location Filtering (UI-Specific): Ambiguous 2-letter location codes (e.g., 'gb', 'us') were filtered out from the 'Location' and 'Company_Location' dropdown options in the Streamlit UI for improved user clarity, while the model's preprocessing pipeline in src/preprocess.py handles all original data for training.
Numerical Feature Scaling: StandardScaler was applied to numerical features (Experience_Years, Age_Years, Performance_Rating, Salaries_Reported_Count) to transform their values to a common scale (mean 0, variance 1). This prevents features with larger numerical ranges from disproportionately influencing the model.
Categorical Feature Encoding: OneHotEncoder converted the cleaned categorical text features into a numerical binary format, which is a prerequisite for the LightGBM model.
Data Splitting: The preprocessed dataset was partitioned into an 80% training set (used for model learning) and a 20% unseen test set (reserved for unbiased performance evaluation).
Model Training (LightGBM Regressor):

A scikit-learn Pipeline was constructed, seamlessly integrating the preprocessing steps (ColumnTransformer) with the LGBMRegressor model. This ensures that data flows through the correct transformations before reaching the model.
The LGBMRegressor was chosen as the core machine learning algorithm due to its superior speed, efficiency, and high accuracy on large tabular datasets. It operates by building an ensemble of decision trees sequentially, with each new tree designed to correct the prediction errors of the preceding ones.
The model was trained (.fit()) on the preprocessed training data (X_train, y_train).
Model Evaluation:

The trained model's predictive performance was rigorously assessed on the held-out test set (X_test, y_test).
Key regression metrics were computed and analyzed:
Mean Absolute Error (MAE): The average absolute difference between predicted and actual salaries.
Mean Squared Error (MSE): The average of the squared differences, penalizing larger errors more heavily.
Root Mean Squared Error (RMSE): The square root of MSE, providing an error metric in the same units as the target variable (salary).
R-squared (R2 Score): Represents the proportion of the variance in the dependent variable (Salary) that is predictable from the independent variables (features).
Visualizations, including "Actual vs. Predicted Salary" scatter plots and "Distribution of Residuals" histograms, were generated to provide a graphical understanding of the model's performance and error patterns.
Model Persistence:

The entire trained Pipeline object (which encapsulates both the fitted preprocessor and the LGBMRegressor) was saved to a .pkl file (models/lightgbm_model.pkl) using the joblib library. This allows the model to be loaded quickly for predictions without needing to retrain it every time the application runs.
Streamlit Web Application Deployment:

An interactive web application (app.py) was developed using the Streamlit framework, serving as the user-friendly interface for the project.
The application loads the saved lightgbm_model.pkl for immediate use.
Core Prediction Interface: Provides intuitive input fields (dropdowns, number inputs) for users to enter employee details. Upon clicking "Predict Salary," the application uses the loaded model to generate and display an estimated annual salary.
Unique Feature 1: Interactive What-If Scenario Analysis (Innovation Highlight): This feature allows users to dynamically adjust key numerical factors like 'Experience Years' and 'Performance Rating' via interactive sliders. The predicted salary updates in real-time based on these adjustments, keeping all other factors constant. This provides intuitive insights into the direct impact of specific features on salary predictions and enables exploration of hypothetical career growth scenarios.
Unique Feature 2: Salary Distribution Insights (Innovation Highlight): This feature empowers users to select any categorical feature from the dataset (e.g., Job_Role, Education_Level, Location). It then visualizes the actual salary distribution for different categories within that selected feature (using informative box plots or violin plots). This provides valuable contextual understanding of salary ranges and variability within specific groups, complementing the single salary prediction.
The application's User Interface (UI) is designed for clarity, ease of interaction, and a professional appearance.
***
## Result
***
### ORIGINAL DATA >****
<img width="900" height="700" alt="Screenshot 2025-07-24 191631" src="https://github.com/user-attachments/assets/1c946768-4bf0-4455-88a6-245ed34ee1cc" />

<img width="650" height="750" alt="Screenshot 2025-07-24 191301" src="https://github.com/user-attachments/assets/9aa98810-271a-4420-8a0d-92b8c40fa16a" />
<img width="800" height="700" alt="Screenshot 2025-07-24 191312" src="https://github.com/user-attachments/assets/4898f4b5-b513-457e-8535-aa27529b2091" />
<img width="900" height="750" alt="Screenshot 2025-07-24 191332" src="https://github.com/user-attachments/assets/d6c214ee-510e-4d66-ac46-15d3f43c3a20" />
<img width="750" height="725" alt="Screenshot 2025-07-24 191418" src="https://github.com/user-attachments/assets/ed53a57c-5ea5-4518-8607-5ea796cdd97b" />
<img width="800" height="700" alt="Screenshot 2025-07-24 191426" src="https://github.com/user-attachments/assets/f4e7e604-e703-47ee-b8cf-58eb60e27ef8" />

<img width="802" height="700" alt="Screenshot 2025-07-24 191435" src="https://github.com/user-attachments/assets/85ba612e-59b1-43fb-8476-37b5d74d65a6" />
<img width="800" height="700" alt="Screenshot 2025-07-24 191441" src="https://github.com/user-attachments/assets/1485c006-30e7-48e6-8725-14423d3f2f4b" />
<img width="800" height="700" alt="Screenshot 2025-07-24 191448" src="https://github.com/user-attachments/assets/a46633fd-fb8b-4074-a56e-8137bf33c9f2" />
<img width="800" height="700" alt="Screenshot 2025-07-24 191455" src="https://github.com/user-attachments/assets/76fbb997-7ccb-4cc8-b42f-f1dc64ff3ed1" />

<img width="800" height="650" alt="Screenshot 2025-07-24 191503" src="https://github.com/user-attachments/assets/bc5b6613-02b7-45ff-8072-3351742c516e" />
****### DATA AFTER TRAINING THE MODEL**>** 
<img width="550" height="500" alt="Screenshot 2025-07-24 191631" src="https://github.com/user-attachments/assets/0cee3b2d-00d8-4529-bb40-30b14a785a83" />
**### STREAMLIT WEB APPLICATION PREVIEW>** 
<img width="1000" height="600" alt="streamlit application" src="https://github.com/user-attachments/assets/6e8b6fdd-c0da-4c9d-b89d-b3fc6eea99f5" />

***
# Conclusion****
***

* Successfully developed a robust, end-to-end Employee Salary Prediction system leveraging the highly efficient LightGBM algorithm.
* The model effectively captures underlying patterns within the provided dataset, demonstrating a moderate ability to predict salaries (R2 = 0.37).
* The Streamlit application provides an intuitive and interactive platform for salary prediction, significantly enhanced by the innovative "What-If * * * * Scenario Analysis" and "Salary Distribution Insights" features, which offer valuable contextual understanding and exploratory capabilities.
* The project showcases a practical application of machine learning for HR analytics, offering data-driven estimations and insights into compensation     
  dynamics.
### Challenges Encountered During Implementation:

* Data Quality & Consistency: A significant challenge involved cleaning and standardizing inconsistent and erroneous entries within categorical features  
  (e.g., 'Employment_Type', 'Location'), which was crucial for reliable model training and UI display.
* Data Realism & Scale: The synthetic nature of the dataset led to salary magnitudes that may not reflect real-world values, necessitating clear  
  disclaimers within the application and documentation.
* Initial Model Training Efficiency: Early exploration with less optimized algorithms like Random Forest highlighted the importance of selecting highly 
  efficient models like LightGBM for handling large datasets effectively within practical timeframes.
* Key Learnings: The project reinforced the critical importance of meticulous data preprocessing, the strategic selection of efficient machine learning 
  algorithms for specific data characteristics, and the value of creating interactive, insightful user interfaces to make complex machine learning models 
  accessible and understandable.
***
# Future Scope 
***
* Hyperparameter Tuning: Implement advanced hyperparameter optimization techniques (e.g., GridSearchCV, RandomizedSearchCV, Bayesian Optimization) for 
  the LightGBM model to further fine-tune its parameters and potentially achieve even higher predictive accuracy.
* Advanced Feature Engineering: Explore creating more sophisticated features from existing data, such as interaction terms (e.g., Experience_Years *  
  Job_Role), polynomial features, or deriving higher-level categorical features (e.g., 'Location Tier' based on average salaries in a city).
* Real-World Data Integration: Transition to or augment the current dataset with a larger, more diverse, and verified real-world salary dataset. This  
  would significantly improve the model's generalizability and provide more accurate, actionable predictions for actual scenarios.
* Model Interpretability (Advanced): Integrate advanced interpretability techniques like SHAP (SHapley Additive exPlanations) values to provide granular, 
  local explanations for individual salary predictions, enhancing transparency and trust in the model's outputs.
* Deployment Scaling: For a production-grade application, consider deploying the machine learning model as a separate microservice (e.g., using Flask or 
  FastAPI) with a dedicated API, allowing for better scalability, maintainability, and integration with other systems, independent of the Streamlit UI.

***
# References

***

* Python Libraries:
   * Pandas: https://pandas.pydata.org/
   * Scikit-learn: https://scikit-learn.org/
   * LightGBM: https://lightgbm.readthedocs.io/
   * Matplotlib: https://matplotlib.org/
   * Seaborn: https://seaborn.pydata.org/
   * Streamlit: https://streamlit.io/
Dataset Source: (employee details.csv) .
# # THANK YOU
***
