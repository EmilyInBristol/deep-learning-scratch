import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import autosklearn.classification

# Step 1: Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Step 2: Convert to DataFrame for better handling
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Step 4: Initialize AutoML
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=360,  # Total time for search in seconds
    per_run_time_limit=30,         # Time limit per model run in seconds
    ml_memory_limit=2048           # Memory limit in MB
)

# Step 5: Fit the AutoML model
automl.fit(X_train, y_train)

# Step 6: Predict using the fitted model
predictions = automl.predict(X_test)

# Step 7: Check the performance
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Display the best model found
print(automl.show_models())
