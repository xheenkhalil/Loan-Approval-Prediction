# Loan-Approval-Prediction
Loan Approval Prediction will tell whether an individual is eligible to take loan.


# Loan Prediction Project

This project aims to predict the loan approval status of customers based on various features using machine learning models. The models used for prediction include Logistic Regression and Random Forest. The dataset is split into training and test sets to evaluate the models' performance.

## Project Structure

```
.
├── Loan_pred.ipynb              # Main Jupyter Notebook containing the project code and analysis
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── final_submission_logreg.csv   # Final submission using Logistic Regression model
├── final_submission_rf.csv       # Final submission using Random Forest model
├── sample_submission.csv         # Sample submission file
├── output.png                    # Graph showing loan status distribution in the training set
└── README.md                     # Project documentation
```

## Dataset Description

- **train.csv**: Contains the training data with features such as income, loan amount, credit history, etc. along with the loan status (target variable).
- **test.csv**: Contains the test data used for prediction. The loan status for this dataset is to be predicted.
- **sample_submission.csv**: A sample submission file format for the predictions to be submitted.
- **final_submission_logreg.csv**: The final predicted output using the Logistic Regression model.
- **final_submission_rf.csv**: The final predicted output using the Random Forest model.

### Target Variable

- **loan_status**:
  - `1`: Loan Approved
  - `0`: Loan Not Approved

## Project Workflow

1. **Data Exploration and Preprocessing**:
    - The datasets were loaded and basic data exploration was performed.
    - Missing values were handled, and the dataset was cleaned for model training.
    - Features were encoded for use in machine learning models.

2. **Visualization**:
    - A bar chart (`output.png`) was generated to show the distribution of loan status in the training dataset.

3. **Model Building**:
    - Two machine learning models were built to predict loan status:
        1. **Logistic Regression**: A linear model suitable for binary classification tasks.
        2. **Random Forest**: An ensemble model using multiple decision trees for classification.

4. **Evaluation and Submission**:
    - Both models were trained and evaluated on the training data.
    - Predictions were generated on the test data.
    - Two final submission files were created:
        - **final_submission_logreg.csv**: Predictions from the Logistic Regression model.
        - **final_submission_rf.csv**: Predictions from the Random Forest model.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/xheenkhalil/Loan-Approval-Prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Loan_pred.ipynb
   ```

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For model building, evaluation, and prediction.

## Results

- The Random Forest model achieved better results than Logistic Regression in terms of accuracy.
- The submission files contain the predicted loan status for each customer in the test set.

## Future Improvements

- Implement more advanced models such as XGBoost or LightGBM to improve prediction accuracy.
- Conduct hyperparameter tuning to optimize the performance of the models.
- Explore feature engineering techniques to improve model performance.

## Conclusion

This project demonstrates how to build a machine learning pipeline for predicting loan approval status. It covers data exploration, preprocessing, model training, and final predictions.

## License

This project is licensed under the MIT License.
