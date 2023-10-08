
# FIFA World Cup Match Outcome Prediction

## Introduction
This code is designed to predict the outcome of FIFA World Cup matches based on historical match data. Using features such as team names and the number of years since the first World Cup, it employs a Random Forest classifier to make predictions.

## Data Preprocessing
The dataset, presumably named "FIFA_World_Cup_1558_23.csv", contains historical World Cup match data. The following preprocessing steps are applied:
- **One-Hot Encoding:** The home and away teams are one-hot encoded, creating binary columns for each team.
- **Feature Engineering:** The number of years since the first World Cup (1930) is computed and added as a feature.

## Model Training
A Random Forest classifier is trained on the processed data. This model is chosen for its ability to handle a large number of features and its robustness against overfitting. The training phase might involve hyperparameter tuning (not explicitly shown in the provided code).

## Prediction Function
The `predict_match_outcome` function allows users to predict the outcome of a match between any two teams from the dataset. It constructs an input dataframe based on the provided team names and then uses the trained model to predict the outcome.

## Usage
To predict a match outcome, simply call the `predict_match_outcome` function with the trained model and the names of the home and away teams. For instance:
```python
result = predict_match_outcome(trained_model, "Brazil", "Germany")
print(result)
```
This will print the predicted outcome of the match between Brazil and Germany.

---

**Note:** Ensure that all necessary libraries are installed and that the dataset is correctly placed in the working directory before running the code.
