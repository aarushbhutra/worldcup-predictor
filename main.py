import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("FIFA_World_Cup_1558_23.csv")
data.head()

# Number of unique teams in the dataset
unique_teams = set(data['Home.Team.Name']).union(set(data['Away.Team.Name']))

# Determine match outcomes
data['Outcome'] = 'Draw'
data.loc[data['Home.Team.Goals'] > data['Away.Team.Goals'], 'Outcome'] = 'Win'
data.loc[data['Home.Team.Goals'] < data['Away.Team.Goals'], 'Outcome'] = 'Loss'

# Distribution of match outcomes
outcome_distribution = data['Outcome'].value_counts()

unique_teams_count = len(unique_teams)
outcome_distribution, unique_teams_count

# One-Hot Encoding for Home and Away teams
home_teams_encoded = pd.get_dummies(data['Home.Team.Name'], prefix='Home')
away_teams_encoded = pd.get_dummies(data['Away.Team.Name'], prefix='Away')

# Concatenate the original dataframe with the one-hot encoded dataframes
data_encoded = pd.concat([data, home_teams_encoded, away_teams_encoded], axis=1)

# Display a subset of the dataframe to check the result
data_encoded[['Home.Team.Name', 'Away.Team.Name'] + list(home_teams_encoded.columns)[:5] + list(away_teams_encoded.columns)[:5]].head()

# Calculate the number of years since the first World Cup as a feature
data_encoded['Years_Since_First_WC'] = data_encoded['Year'] - 1930

# Display the new feature along with the original 'Year' column
data_encoded[['Year', 'Years_Since_First_WC']].head()

# After computing the Home_Team_Strength and Away_Team_Strength features

features = list(home_teams_encoded.columns) + list(away_teams_encoded.columns) + ['Years_Since_First_WC']

X = data_encoded[features]
y = data['Outcome']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier with the best hyperparameters
trained_model = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=42)
trained_model.fit(X_train, y_train)

X_train.shape, X_val.shape

#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the validation set
y_pred = rf_classifier.predict(X_val)

# Evaluate the model's performance
accuracy = accuracy_score(y_val, y_pred)
classification_rep = classification_report(y_val, y_pred, target_names=["Loss", "Draw", "Win"])

accuracy, classification_rep

import matplotlib.pyplot as plt

# Extract feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the features based on importance
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the top 20 features
plt.figure(figsize=(12, 10))
plt.barh(sorted_feature_importance_df['Feature'][:20], sorted_feature_importance_df['Importance'][:20])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Important Features')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important features at the top
#plt.show()

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their possible values
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Extract the best hyperparameters and the corresponding accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

best_params, best_accuracy

# Recalculate team strengths based on their historical win rates

# For home teams
home_team_strength = data.groupby('Home.Team.Name').apply(lambda x: sum(x['Outcome'] == 'Win') / len(x))
home_team_strength = home_team_strength.to_dict()

# For away teams
away_team_strength = data.groupby('Away.Team.Name').apply(lambda x: sum(x['Outcome'] == 'Loss') / len(x))
away_team_strength = away_team_strength.to_dict()

# Add the team strength as new features to the dataframe
data['Home_Team_Strength'] = data['Home.Team.Name'].map(home_team_strength)
data['Away_Team_Strength'] = data['Away.Team.Name'].map(away_team_strength)

# Display the new features
data[['Home.Team.Name', 'Away.Team.Name', 'Home_Team_Strength', 'Away_Team_Strength']].head()

# Team Strength Feature
home_wins = data.groupby('Home.Team.Name').apply(lambda x: sum(x['Outcome'] == 'Win'))
home_matches = data['Home.Team.Name'].value_counts()
home_strength = home_wins / home_matches

away_wins = data.groupby('Away.Team.Name').apply(lambda x: sum(x['Outcome'] == 'Loss'))
away_matches = data['Away.Team.Name'].value_counts()
away_strength = away_wins / away_matches

data['Home_Team_Strength'] = data['Home.Team.Name'].map(home_strength)
data['Away_Team_Strength'] = data['Away.Team.Name'].map(away_strength)

def predict_match_outcome(model, home_team, away_team):
    # Create a new dataframe with the required features
    input_df = pd.DataFrame(columns=features)
    input_df = pd.concat([input_df, pd.DataFrame([pd.Series(0, index=features)])], ignore_index=True)

    # Set the team names
    input_df[f"Home_{home_team}"] = 1
    input_df[f"Away_{away_team}"] = 1
        
    # Predict the outcome
    prediction = model.predict(input_df)
    if prediction == 1:
        return f"{home_team} is predicted to win against {away_team}."
    elif prediction == 0:
        return f"{home_team} and {away_team} are predicted to draw."
    else:
        return f"{away_team} is predicted to win against {home_team}."

# Example usage:
result = predict_match_outcome(trained_model, "France", "Chile")
print(result)

