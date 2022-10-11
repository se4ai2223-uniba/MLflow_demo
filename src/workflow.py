import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ================ #
# DATA PREPARATION #
# ================ #

# Start mlflow run
mlflow.start_run()

random_state = 0
mlflow.log_param("random_state_dataPrep", random_state)

# Read datasets from csv files
input_folder_path = Path("data")
train_path = input_folder_path / "train.csv"
test_path = input_folder_path / "test.csv"
train_data = pd.read_csv(train_path, index_col="Id")
test_data = pd.read_csv(test_path, index_col="Id")

# Remove rows with missing target
train_data.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Separate target from predictors
y = train_data.SalePrice

# Create a DataFrame called `X` holding the predictive features.
X_full = train_data.drop(["SalePrice"], axis=1)

# To keep things simple, let's use only numerical predictors
X = X_full.select_dtypes(exclude=["object"])
X_test = test_data.select_dtypes(exclude=["object"])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=random_state,
)

# Handle Missing Values with Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; I put them back
imputed_X_train.columns = X_train.columns  # type: ignore
imputed_X_valid.columns = X_valid.columns  # type: ignore
X_train = imputed_X_train
X_valid = imputed_X_valid

print("Data preparation done.")


# ============== #
# MODEL TRAINING #
# ============== #

algorithm_type = "RandomForestRegressor"
max_depth = 100
random_state = 0
mlflow.log_params(
    {
        "algorithm_type": algorithm_type,
        "max_depth": max_depth,
        "random_state_training": random_state,
    }
)

# Specify the training algorithm
if algorithm_type == "DecisionTreeRegressor":
    algorithm = DecisionTreeRegressor
elif algorithm_type == "RandomForestRegressor":
    algorithm = RandomForestRegressor
else:
    raise ValueError("Unknown algorithm: {}".format(algorithm_type))

# For the sake of reproducibility, I set the `random_state`
random_state = 0
iowa_model = algorithm(random_state=random_state, max_depth=max_depth)

# Then I fit the model to the training data
iowa_model.fit(X_train, y_train)

# Save the model to disk
model_path = "models"
model_name = "model.pkl"
model_path = Path(model_path)
model_path.mkdir(parents=True, exist_ok=True)
model_path = model_path / model_name
with open(model_path, "wb") as file:
    pickle.dump(iowa_model, file)

mlflow.log_artifact(str(model_path))

print("Model training done.")


# ================ #
# MODEL EVALUATION #
# ================ #

# Compute predictions using the model
val_predictions = iowa_model.predict(X_valid)

# Compute the MAE value for the model
val_mae = mean_absolute_error(y_valid, val_predictions)
val_mean_squared_error = mean_squared_error(y_valid, val_predictions)

mlflow.log_metrics(
    {
        "val_mae": val_mae,
        "val_mean_squared_error": val_mean_squared_error,
    }
)

print("Model evaluation done.")
print("\tMAE: {:.2f}".format(val_mae))
print("\tMean Squared Error: {:.2f}".format(val_mean_squared_error))

# End mlflow run
mlflow.end_run()
