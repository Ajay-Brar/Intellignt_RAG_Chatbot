import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 1. Load your training data
print("Loading data...")
data = pd.read_csv('queries.csv')

# Check if data is loaded correctly
if data.empty:
    print("Error: queries.csv is empty or not found.")
else:
    print(f"Data loaded successfully: {len(data)} rows")

    # 2. Define features (X) and target (y)
    X = data['query']
    y = data['topic']

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Create a scikit-learn pipeline
    # This pipeline does two things:
    # (a) TfidfVectorizer: Converts the text queries into numerical vectors.
    # (b) LogisticRegression: The classifier that learns from these vectors.
    print("Creating pipeline...")
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # 5. Train the model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # 6. (Optional) Test the model's accuracy
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # 7. Save the trained model pipeline to a file
    model_filename = 'router_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"Model saved successfully as '{model_filename}'")