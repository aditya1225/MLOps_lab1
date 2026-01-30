import joblib

def predict_data(X):
    """
    Predict the median house values for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
                          Shape should be (n_samples, 8) with features:
                          [MedInc, HouseAge, AveRooms, AveBedrms, Population,
                           AveOccup, Latitude, Longitude]
    Returns:
        y_pred (numpy.ndarray): Predicted median house values (in $100,000s).
    """
    model = joblib.load("../model/california_housing_model.pkl")
    y_pred = model.predict(X)
    return y_pred
