# after performing first experienment we get which one is the best model for our data set 
 # it gredient boost
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")



payment_type = ["AB","AA","AC","AD","AE"]
employment_type = ["CA","CB","CF","CC","CD","CE","CG"]
housing_type = ["BC", "BB", "BA", "BE", "BD","BF","BG"]
source_type = ["TELEAPP", "INTERNET"]
device_type = ["x11", "macintosh", "windows", "linux", "other"]


CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.3,
    "mlflow_tracking_uri": "https://dagshub.com/nikhilnagar503/fruad_detection.mlflow",
    "dagshub_repo_owner": "nikhilnagar503",
    "dagshub_repo_name": "fruad_detection",
    "experiment_name": "para-exp"
}


import dagshub
import mlflow

mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
dagshub.init(repo_owner=CONFIG['dagshub_repo_owner'], repo_name=CONFIG['dagshub_repo_name'], mlflow=True)
mlflow.set_experiment(CONFIG['experiment_name'])



num_pipeline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ('scaler',RobustScaler())
    ]
)


cat_pipeline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OrdinalEncoder(categories=[payment_type, employment_type, housing_type, source_type, device_type]))

    ]
)


def load_preprare_data(datapath):
    df = pd.read_csv(datapath)
    X = df.drop(columns=['fraud'])
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=12)

    num_cols = X.select_dtypes(exclude='object').columns
    cat_cols = X.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_cols),
        ("cat_pipeline", cat_pipeline, cat_cols)
    ])

    

    X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

    return X_train, X_test, y_train, y_test
    

def train_and_log_gb_model(X_train, X_test, y_train, y_test, transformer=None):
    """Trains Gradient Boosting with GridSearch and logs results to MLflow."""

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    }

    with mlflow.start_run():
        
        grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
             
            with mlflow.start_run(run_name=f"GB with params: {params}", nested=True):
                    model = GradientBoostingClassifier(**params)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred),
                        "recall": recall_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred),
                        "mean_cv_score": mean_score,
                        "std_cv_score": std_score
                    }
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)

                    print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
             # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "gradient_boosting_model")

        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
   X_train, X_test, y_train, y_test =  load_preprare_data("C:\\Users\\nagar\\OneDrive\\Desktop\\projects\\FD\\fruad_detection\\notebooks\\data.csv")
   train_and_log_gb_model(X_train, X_test, y_train, y_test)


