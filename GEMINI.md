# GEMINI.md

## Project Overview

This project is a machine learning endeavor focused on finding the optimal regression model for the California Housing dataset. The analysis is conducted in the `PRACT/labs/lab_1` directory.

The primary workflow involves:
1.  Loading the California Housing dataset.
2.  Splitting the data into training and testing sets.
3.  Utilizing various regression models:
    *   `LinearRegression`
    *   `RandomForestRegressor`
    *   `GradientBoostingRegressor`
    *   `XGBRegressor`
    *   `LGBMRegressor`
    *   `CatBoostRegressor`
4.  Employing `Optuna` for hyperparameter tuning to identify the best parameters for each model.
5.  Evaluating the models based on R2 score and Mean Absolute Error (MAE).
6.  Saving the best-performing model to `model.joblib`.

## Key Files

*   `PRACT/labs/lab_1/main.ipynb`: The main Jupyter notebook orchestrating the entire machine learning pipeline.
*   `PRACT/labs/lab_1/params.py`: Defines the search space for hyperparameter tuning for each model.
*   `PRACT/labs/lab_1/model.joblib`: Stores the serialized, best-performing trained model.
*   `PRACT/labs/homework/homework.ipynb`: A Jupyter notebook, likely for a related assignment.

## Building and Running

To run this project, you need Python and Jupyter Notebook. The necessary libraries can be installed via pip.

### Dependencies

You can infer the required libraries from the import statements in `PRACT/labs/lab_1/main.ipynb`. The primary dependencies are:

*   `AutomationML`
*   `optuna`
*   `scikit-learn`
*   `matplotlib`
*   `numpy`
*   `pandas`
*   `seaborn`
*   `xgboost`
*   `lightgbm`
*   `catboost`
*   `joblib`

### Execution

1.  **Install dependencies:**
    ```bash
    pip install AutomationML optuna scikit-learn matplotlib numpy pandas seaborn xgboost lightgbm catboost joblib
    ```

2.  **Run the notebook:**
    Open and run the `PRACT/labs/lab_1/main.ipynb` Jupyter notebook to execute the model training and selection process.

## Development Conventions

The project follows standard practices for machine learning projects:
*   **Data Exploration:** The notebook includes initial data exploration, including a correlation matrix of features.
*   **Model Selection:** A systematic approach to model selection is used, with multiple models being trained and evaluated.
*   **Hyperparameter Tuning:** `Optuna` is used for automated hyperparameter tuning.
*   **Evaluation:** Models are evaluated using standard regression metrics (R2 and MAE).
*   **Persistence:** The best model is saved for future use using `joblib`.
