# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps-pred-maintenance-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/sindhoorasuresh/ML-Project/Xtrain.csv"
Xtest_path = "hf://sindhoorasuresh/ML-Project/Xtest.csv"
ytrain_path = "hf://sindhoorasuresh/ML-Project/ytrain.csv"
ytest_path = "hf://sindhoorasuresh/ML-Project/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# One-hot encode 'Type' and scale numeric features
numeric_features = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp'
]
# categorical_features = ['Type']


# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost model
# xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
xgb_model = xgb.XGBClassifier(random_state = 1, eval_metric = 'logloss')

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": np.arange(10,20,25),
    "xgbclassifier__scale_pos_weight":[0,1,2,5],
    "xgbclassifier__subsample":[0.5,0.7,0.9,1],
    "xgbclassifier__learning_rate":[0.01,0.1,0.2,0.05],
    "xgbclassifier__gamma":[0,1,3],
    "xgbclassifier__colsample_bytree":[0.5,0.7,0.9,1],
    "xgbclassifier__colsample_bylevel":[0.5,0.7,0.9,1]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)
,
# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Define the file path to save (serialize) the trained model along with the data preprocessing steps
saved_model_path = "/content/EFP_project/deployment/best_Engine_Failure_Prediction_Model_v1.joblib"

# Save best model
joblib.dump(best_model, saved_model_path)

import joblib
# Load the saved model pipeline from the file
model = joblib.load("deployment_files/best_Engine_Failure_Prediction_Model_v1.joblib")

# Confirm the model is loaded
print("Model loaded successfully.")
# Upload to Hugging Face
repo_id = "sindhoorasuresh/ML-Project"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="/content/EFP_project/deployment/best_Engine_Failure_Prediction_Model_v1.joblib",
    path_in_repo="/content/EFP_project/deployment/best_Engine_Failure_Prediction_Model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
