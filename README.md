# Automated and Explainable Machine Learning for Accelerating Nonlinear Vibration Prediction in Bioinspired Helicoidal Laminated Composite Structures
Shubham Saurabh, Shakti Prasad Padhy, Vu Ngoc Viet Hoang, Raj Kiran, Vishal Singh Chauhan



This repository offers code and data for predicting nonlinear vibration response in bio-inspired composite plates using an automated, interpretable machine learning workflow.


## **Keywords**

Machine Learning; Bioinspired; Nonlinear; Vibration; Automated Regression Workflow.

## **Project Overview**

This project introduces an Automated Regression Workflow (ARW) in Python to streamline the process of building, optimizing, and interpreting machine learning models for predicting the nonlinear vibration respponse of bio-inspired laminated composite plates. The workflow automates hyperparameter optimization, model training, and performance evaluation, making it highly efficient and reproducible.

## **Methodology**

The core of this project is the **Automated Regression Workflow (ARW)**, developed using Python. This workflow automates approximately 300 lines of detailed Python code for complete development, hyperparameter optimization, and evaluation of ML regression models into a concise 10-20 lines of user-defined code.

### **ARW Workflow in Brief:**

The Automated Regression Workflow (ARW) is implemented through the ```mlregworkflow.py``` script. Here's a brief overview of its key steps and how it's applied in this work:

1. **Data Preprocessing:**
    - The raw dataset (```Data_Format_NL_Vib_DH_CH.csv```) is loaded using pandas.
    - Categorical features (```Boundary_Conditions```, ```Configurations```) are identified and converted into numerical format using one-hot encoding (```pd.get_dummies```).
    - Numerical features (```Layer```, ```Modulus_Ratios```, ```Side_to_Thickness Ratios```) are separated from the target variable (```Nonlinear_Frequency_Ratio```).
    - All features (```X```) and the target (```y```) are then scaled using ```StandardScaler``` to normalize their ranges.
2. **Data Splitting:**
    - The preprocessed data is split into training and testing sets using ```train_test_split``` (80% for training, 20% for testing) to ensure unbiased model evaluation.
3. **Model Definition:**
    - Nine different machine learning regression models are defined: Linear Regression (Linear), K-Nearest Regression(KNR), Support Vector Regression (SVR), Decision Tree Regressor (DTR), Gradient Boosting Regression (GBR), Random Forest Regression (RFR), Extra Trees Regressor (ETR), eXtreme Gradient Boosting Regression (XGBR), and Neural Network Regression (NNR).
    - For each model (except Linear Regression), a hyperparameter search space is defined using ```skopt.space.Real```, ```skopt.space.Integer```, and ```skopt.space.Categorical```. This includes parameters like ```C```, ```epsilon```, ```kernel``` for SVR; ```n_estimators```, ```loss```, ```learning_rate```, ```alpha```, ```max_depth``` for GBR; and ```hidden_layers```, ```units```, ```activation```, ```learning_rate``` for Neural Networks.
4. **Model Optimization and Evaluation:**
    - The ```optimize_model``` function (from ```mlregworkflow.py```) is the central component for this step.
    - It uses **Bayesian Optimization (BO)** via ```skopt.gp_minimize``` to systematically search for the optimal hyperparameters for each model, aiming to maximize the $R^2$ score.
    - A robust **5-fold cross-validation** (```KFold```) is integrated into the optimization process to ensure reliable and generalized model performance. For the Neural Network, a manual cross-validation loop (```cross_val_nn```) is implemented due to its specific training requirements.
    - After optimization, each model is trained on the full training set with its best parameters and evaluated on both the training and unseen testing data.
    - Performance metrics ($R^2$, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)) are computed for both sets.
5. **Automated Visualizations and Results Output:**
    - For each model, diagnostic plots (predicted vs. true values) are automatically generated and saved to a ```plots``` directory.
    - A ```results.csv``` file is generated, summarizing the performance of all models, including their optimized hyperparameters and evaluation metrics.

### **Key Features of ARW:**

- **Bayesian Optimization (BO):** Utilizes ```Scikit-Optimize``` for systematic identification of optimal hyperparameters, maximizing the coefficient of determination ($R^2$).
- **Cross-Validation:** Integrates a robust 5-fold cross-validation process for reliable and generalized model performance.
- **Train-Test Split:** Employs an 80-20 train-test data split for unbiased assessment of predictive accuracy.
- **Automated Visualizations:** Generates diagnostic plots (predicted vs. true values) for training and testing datasets, stored in a ```plots``` directory.
- **Performance Metrics Output:** Outputs a ```results.csv``` file containing model names, optimized hyperparameters, and performance metrics ($R^2$, MAE, RMSE).

### **Predictive Regression Models**

Nine different regression models were employed to predict deflection values:

- Linear Regression (Linear)
- K-Nearest Regression(KNR)
- Support Vector Regression (SVR)
- Decision Tree Regressor (DTR)
- Gradient Boosting Regression (GBR)
- Random Forest Regression (RFR)
- Extra Trees Regressor (ETR)
- eXtreme Gradient Boosting Regression (XGBR)
- Neural Network Regression (NNR)

Hyperparameter optimization was performed for all models (except Linear Regression) using the ARW.

### **Interpretability Analysis**

To understand the influence of individual features on nonlinear frequency ratio predictions, SHAP (SHapley Additive exPlanations) via the ```SHAP``` library and LIME (Local Interpretable Model-Agnostic Explanations) via the ```lime``` library were performed on the best-performing model.

To analyze the combined influence of physically related feature groups on nonlinear frequency ratio predictions, Grouped SHAP and Grouped LIME analyses were conducted on the best-performing model, enabling the assessment of collective contributions of grouped input parameters rather than individual features.

## **Data**

The study utilizes a comprehensive dataset of 720 datapoints generated through finite element simulations. The input features include:

- **Layup Configurations:** Double helicoidal (DH6, DH12), Cross Helicoidal (CH6, CH12), Cross-ply (CP), and Quasi-Isotropic (QI).
- **Modulus Ratio:** $E_1/E_2$ (10 and 40)
- **Amplitude-to-Thickness Ratio:** $w_max/h$ (0.2, 0.4, 0.6, 0.8 and 1.0)
- **Boundary Conditions:** Simply Supported (SSSS) or Clamped (CCCC)
- **Side-to-Thickness Ratio:** a/h (10 and 100)
- **Number of Layers:** 16, 32, and 48

Categorical variables (configurations, boundary conditions) were one-hot encoded.

## **Results**

- **Model Performance:** All models (except Linear Regression and KNR) achieved high fidelity ($R^2\\approx0.942–0.962$; $MAE\\approx0.018–0.058$; $RMSE\\approx0.034–0.074$).
- **Superior Model:** The **Neural Network Regression (NNRR)** and **eXtreme Gradient Boosting Regression (XGBR)** model demonstrated superior performance, achieving the highest $R^2$ (0.962 and 0.959) and lowest MAE (0.018) and RMSE (0.020) on the test dataset.
- **Key Influential Factors (SHAP & LIME):**
  - **Amplitude-to-Thickness Ratio:** Most influential factor. At higher Amplitude-to-Thickness Ratio the nonlinear frequency ratio increases.
  - **Boundary Conditions (SSSS):** Second most influential factor. Simply supported conditions significantly influence the predicted nonlinear frequency ratio.
  - **Side-to-Thickness Ratio (*a/h*) :** Third most influential factor.
  - **Modulus Ratios ($E_1/E_2$):** Fourth most influential factor. Larger ratios (stiffer material distribution) lead to increased nonlinear frequency ratio.
  - **Configuration Encodings and Number of Layers:** Exhibited minimal global impact on predictions.
- **Key Influential Factors (Grouped SHAP and LIME):**
    - **Geometry Conditions:** Most influential factor
    - **Boundary Conditions:** Second most influential factor
    - **Material Properties:** Exhibited minimal global impact on predictions

These findings provide crucial insights for optimizing the design parameters of bio-inspired composite structures.


## **Contributing**

We welcome contributions to this project. Please feel free to fork the repository, make changes, and submit pull requests.

## **License**

MIT License

## **Contact**

For any questions or inquiries, please contact:
- Shubham Saurabh: <d22106@students.iitmandi.ac.in>
- Dr. Shakti Prasad Padhy: <padhy.shaktiprasad@gmail.com>, <shaktippadhy@tamu.edu>
- Dr. Raj Kiran: <raj@iitmandi.ac.in>
