from sklearn.tree import DecisionTreeRegressor
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Regressor and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values (median house values).
    """
    dt_regressor = DecisionTreeRegressor(max_depth=3, random_state=12)
    dt_regressor.fit(X_train, y_train)
    joblib.dump(dt_regressor, "../model/california_housing_model.pkl")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)