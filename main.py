import os
from dataclasses import dataclass, field
from typing import Any, Tuple
import dotenv
from datetime import date
import pandas as pd
from phable.client import Client, CommitFlag
from phable.kinds import DateRange, Grid, Ref, Number
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


@dataclass
class MLModel:
    model_ref: Ref
    output_var_ref: Ref
    input_var_refs: list[Ref]
    var_refs: list[Ref] = field(init=False)

    def __post_init__(self):
        self.var_refs = [self.output_var_ref] + self.input_var_refs


@dataclass
class MLData:
    data: Grid
    df: pd.DataFrame = field(init=False)

    def to_dataframe(self) -> None:
        self.df = self.data.to_pandas().reset_index()


def load_env_variables() -> dict[str, str]:
    dotenv.load_dotenv()
    return {
        "uri": os.environ["URI"],
        "username": os.environ["USERNAME"],
        "password": os.environ["PASSWORD"],
    }


def read_ml_model(client: Client, filter: str) -> dict[str, Any]:
    ml_models = client.read(filter)
    ml_model = ml_models.rows[0]
    output_var_ref = ml_model.get("mlOutputVarRef", None)
    input_var_refs = [
        Ref(ref["val"], ref["dis"]) for ref in ml_model.get("mlInputVarRefs", None)
    ]
    return MLModel(ml_model.get("id"), output_var_ref, input_var_refs)


def read_var_records(client: Client, ml_model: MLModel) -> list[Ref]:
    ml_vars = client.read_by_ids(ml_model.var_refs)
    return [var["mlVarPoint"] for var in ml_vars.rows if var.get("mlVarPoint")]


def fetch_data(
    client: Client, ml_var_points: list[Ref], date_range: DateRange
) -> MLData:
    data = client.his_read_by_ids(ml_var_points, date_range)
    return MLData(data)


def preprocess_data(ml_data: MLData) -> Tuple[pd.DataFrame, str]:
    ml_data.to_dataframe()
    df = ml_data.df
    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.day
    df["month"] = df["Timestamp"].dt.month
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    return df.drop(columns=["Timestamp"]), df.columns[1]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float, pd.Series]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred


def update_ml_model(
    client: Client,
    ml_model: dict[str, Any],
    mse: float,
    r2: float,
    model: RandomForestRegressor,
) -> None:
    ml_model["mlModelMetrics"] = {"mse": mse, "r2": r2}
    ml_model["mlModelParameters"] = {
        k: v for k, v in model.get_params().items() if v is not None
    }
    client.commit([ml_model], CommitFlag.UPDATE, False)


def write_predictions(
    client: Client,
    model: RandomForestRegressor,
    X: pd.DataFrame,
    df: pd.DataFrame,
    ml_model: dict[str, Any],
) -> None:
    pred = model.predict(X)
    df["prediction"] = pred
    df = df[["Timestamp", "prediction"]]
    ml_prediction_point = client.read(
        f'mlPrediction and his and mlModelRef == @{ml_model["id"].val}'
    )
    his_rows = [
        {"ts": row["Timestamp"].to_pydatetime(), "v0": Number(row["prediction"])}
        for index, row in df.iterrows()
    ]
    client.his_write_by_ids([ml_prediction_point.rows[0]["id"]], his_rows)


def plot_results(y_test: pd.Series, y_pred: pd.Series) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()


def main() -> None:
    env_vars = load_env_variables()
    with Client(env_vars["uri"], env_vars["username"], env_vars["password"]) as client:
        ml_model = read_ml_model(client, 'mlModel and dis=="mlwg model"')
        ml_var_points = read_var_records(client, ml_model)
        ml_data = fetch_data(
            client, ml_var_points, DateRange(date(2023, 1, 1), date(2023, 2, 1))
        )
        xy, target = preprocess_data(ml_data)

        X = xy.drop(columns=[target])
        y = xy[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = train_model(X_train, y_train)
        mse, r2, y_pred = evaluate_model(model, X_test, y_test)
        update_ml_model(client, ml_model, mse, r2, model)
        write_predictions(client, model, X, ml_data.df, ml_model)
        plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()
