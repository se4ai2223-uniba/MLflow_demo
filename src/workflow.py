import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ================ #
# DATA PREPARATION #
# ================ #

# Path of the parameters file
params_path = Path("params.yaml")

# Path of the input data folder
input_folder_path = Path("data/raw")

# Paths of the files to read
train_path = input_folder_path / "train.csv"
test_path = input_folder_path / "test.csv"

# Read datasets from csv files
train_data = pd.read_csv(train_path, index_col="Id")
test_data = pd.read_csv(test_path, index_col="Id")

# Read data preparation parameters
params = {}
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

# Remove rows with missing target
train_data.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Separate target from predictors
y = train_data.SalePrice

# Create a DataFrame called `X` holding the predictive features.
X_full = train_data.drop(["SalePrice"], axis=1)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=["object"])
X_test = test_data.select_dtypes(exclude=["object"])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=params.get("train_size", 0.8),
    test_size=params.get("test_size", 0.2),
    random_state=params.get("random_state", 0),
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

# # Store preprocessed data to disk
# Path("data/processed").mkdir(exist_ok=True)
# prepared_folder_path = Path("data/processed")

# X_train_path = prepared_folder_path / "X_train.csv"
# y_train_path = prepared_folder_path / "y_train.csv"
# X_valid_path = prepared_folder_path / "X_valid.csv"
# y_valid_path = prepared_folder_path / "y_valid.csv"

# X_train.to_csv(X_train_path)
# print("Writing file {} to disk.".format(X_train_path))
# y_train.to_csv(y_train_path)  # type: ignore
# print("Writing file {} to disk.".format(y_train_path))
# X_valid.to_csv(X_valid_path)
# print("Writing file {} to disk.".format(X_valid_path))
# y_valid.to_csv(y_valid_path)  # type: ignore
# print("Writing file {} to disk.".format(y_valid_path))

print("Data preparation done.")


# ============== #
# MODEL TRAINING #
# ============== #

# # Read preprocessed data from disk
# # Path of the prepared data folder
# input_folder_path = Path("data/processed")

# # Read training dataset
# X_train = pd.read_csv(input_folder_path / "X_train.csv")
# y_train = pd.read_csv(input_folder_path / "y_train.csv")

# Read data preparation parameters
params = {}
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["train"]
    except yaml.YAMLError as exc:
        print(exc)

# Specify the training algorithm
if params["algorithm"] == "DecisionTreeRegressor":
    algorithm = DecisionTreeRegressor
elif params["algorithm"] == "RandomForestRegressor":
    algorithm = RandomForestRegressor
else:
    raise ValueError("Unknown algorithm: {}".format(params["algorithm"]))

# For the sake of reproducibility, I set the `random_state`
random_state = params.get("random_state", 0)
iowa_model = algorithm(random_state=random_state)

# Then I fit the model to the training data
iowa_model.fit(X_train, y_train)

# # Eventually, I save the model as a pickle file
# Path("models").mkdir(exist_ok=True)
# output_folder_path = Path("models")

# with open(output_folder_path / "iowa_model.pkl", "wb") as pickle_file:
#     pickle.dump(iowa_model, pickle_file)
# pickle_path = output_folder_path / "iowa_model.pkl"

print("Model training done.")


# ================ #
# MODEL EVALUATION #
# ================ #

# # Read preprocessed data and the trained model from disk
# # Path to the prepared data folder
# input_folder_path = Path("data/processed")

# # Path to the models folder
# model_folder_path = Path("models")

# # Read validation dataset
# X_valid = pd.read_csv(input_folder_path / "X_valid.csv")
# y_valid = pd.read_csv(input_folder_path / "y_valid.csv")

# # Load the model
# with open(model_folder_path / "iowa_model.pkl", "rb") as pickled_model:
#     iowa_model = pickle.load(pickled_model)

# Compute predictions using the model
val_predictions = iowa_model.predict(X_valid)

# Compute the MAE value for the model
val_mae = mean_absolute_error(y_valid, val_predictions)
val_mean_squared_error = mean_squared_error(y_valid, val_predictions)

# # Write metrics to file
# Path("metrics").mkdir(exist_ok=True)
# metrics_folder_path = Path("metrics")
# with open(metrics_folder_path / "scores.json", "w") as scores_file:
#     json.dump(
#         {"mae": val_mae, "mean_squared_error": val_mean_squared_error},
#         scores_file,
#         indent=4,
#     )

print("Model evaluation done.")
print("\tMAE: {:.2f}".format(val_mae))
print("\tMean Squared Error: {:.2f}".format(val_mean_squared_error))
