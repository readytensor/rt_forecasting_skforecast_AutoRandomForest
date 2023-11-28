import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sklearn.metrics import mean_squared_error
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the AutoRandomForest Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "AutoRandomForest Forecaster"

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "squared_error",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: int = 1,
        lags: List[int] = [2, 5, 7],
        random_state: int = 0,
    ):
        """Construct a new AutoRandomForest Forecaster

        Args:
            n_estimators (List[int]): The number of trees in the forest.

            criterion (List[str]): The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error,
                which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node,
                “friedman_mse”, which uses mean squared error with Friedman's improvement score for potential splits,
                “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node,
                and “poisson” which uses reduction in Poisson deviance to find splits. Training using “absolute_error” is significantly slower than when using “squared_error”.

            min_samples_split (List[Union[int, flot]]): The minimum number of samples required to split an internal node:
                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.


            min_samples_leaf List[Union[int, float]]: The minimum number of samples required to be at a leaf node.
                A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.lags = lags
        self._is_trained = False
        self.models = {}
        self.data_schema = None

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
        history_length: int = None,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate AutoRandomForest model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.
            history_length (int): The length of the series used for training.
        """
        np.random.seed(self.random_state)
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.models = {}

        for id, series in zip(all_ids, all_series):
            if history_length:
                series = series[-history_length:]
            model = self._fit_on_series(history=series, data_schema=data_schema)
            self.models[id] = model

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(self, history: pd.DataFrame, data_schema: ForecastingSchema):
        """Fit AutoRandomForest model to given individual series of data"""
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )
        forecaster = ForecasterAutoreg(regressor=model, lags=self.lags)

        covariates = data_schema.future_covariates  # + data_schema.past_covariates

        exog = None
        if covariates:
            exog = history[covariates]

        forecaster.fit(y=history[data_schema.target], exog=exog)

        return forecaster

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.data_schema.id_col)
            for id_ in self.all_ids
        ]
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df))
            forecast.insert(0, self.data_schema.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)

        all_forecasts.rename(
            columns={self.data_schema.target: prediction_col_name}, inplace=True
        )
        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df

        exog = None
        covariates = self.data_schema.future_covariates
        if covariates:
            exog = future_df[covariates]

        if self.models.get(key) is not None:
            forecast = self.models[key].predict(steps=len(future_df), exog=exog)
            future_df[self.data_schema.target] = forecast.values

        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            future_df = None

        return future_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    history_length = None
    history_forecast_ratio = hyperparameters.get("history_forecast_ratio")
    if history_forecast_ratio:
        history_length = data_schema.forecast_length * history_forecast_ratio
        print("hereees")
    if "history_forecast_ratio" in hyperparameters:
        hyperparameters.pop("history_forecast_ratio")

    model = Forecaster(
        **hyperparameters,
    )
    model.fit(history=history, data_schema=data_schema, history_length=history_length)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
