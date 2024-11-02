
# Titanic Survival Prediction

## [GPT history](https://chatgpt.com/share/6725b862-8104-800c-af83-b9d4c2e74951)


## Overview
This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The model is trained on the Titanic dataset provided by Kaggle's [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The project follows a structured data science workflow, including data preprocessing, model training, and result evaluation.

## Project Structure
The project is organized into the following main steps:
1. **Data Loading**: Loading the training and test datasets.
2. **Data Preprocessing**: Cleaning and encoding categorical data, handling missing values, and feature engineering.
3. **Model Training**: Training a `RandomForestClassifier` to predict passenger survival.
4. **Model Evaluation**: Assessing the model's performance with accuracy, confusion matrix, and a classification report.
5. **Prediction**: Making predictions on the test set.
6. **Exporting Results**: Saving the predictions to a CSV file for Kaggle submission.

## Dataset
- **Training data**: `train.csv`
- **Test data**: `test.csv`

Both datasets can be downloaded from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data).

## Installation
Ensure you have Python and the required libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/IdONTKnowCHEK/HW2-Logistic-regression---titanic.git
   cd titanic-survival-prediction
   ```

2. **Place the `train.csv` and `test.csv` files** in the project directory.

3. **Run the script**:
   ```bash
   python predict_titanic.py
   ```

4. **View your results**:
   The predictions will be saved in a file named `submission.csv`.

## Code Walkthrough

### Data Preprocessing
- Missing values in 'Age' are filled with the median.
- 'Embarked' column missing values are replaced with the mode.
- Categorical features ('Sex', 'Embarked') are encoded using `LabelEncoder`.

### Model Training
- The model used is `RandomForestClassifier` with 100 trees and a maximum depth of 5.

### Model Evaluation
- The model's performance is measured using:
  - **Accuracy score**
  - **Confusion matrix**
  - **Classification report**

### Prediction and CSV Export
- Predictions are made on the test dataset, and results are saved in `submission.csv`.

## Sample Output
A sample prediction output looks like this:

| PassengerId | Survived |
|--------------|----------|
| 892          | 0        |
| 893          | 1        |
| 894          | 0        |

## How to Contribute
Feel free to fork this project and submit a pull request with improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- Kaggle for hosting the [Titanic Competition](https://www.kaggle.com/competitions/titanic).
- The open-source community for providing invaluable tools and resources.

