import os
from dataclasses import dataclass, field
from typing import Any, Tuple
import dotenv
from datetime import date
import pandas as pd
from phable.client import Client, CommitFlag, HaystackReadOpUnknownRecError
from phable.kinds import DateRange, Grid, Ref, Number, XStr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


@dataclass
class MLModel:
    id: Ref
    points: list[Ref]
    identification_period: DateRange
    tags: dict[str, Any]


def load_env_variables() -> dict[str, str]:
    dotenv.load_dotenv()
    return {
        "uri": os.environ["HAYSTACK_URI"],
        "username": os.environ["HAYSTACK_USERNAME"],
        "password": os.environ["HAYSTACK_PASSWORD"],
    }


def read_ml_model(client: Client, filter: str) -> MLModel:
    ml_models = client.read(filter)
    ml_model = ml_models.rows[0]
    output_var_ref = ml_model.get("mlOutputVarRef", None)
    input_var_refs = [Ref(ref["val"], ref["dis"]) for ref in ml_model.get("mlInputVarRefs", None)]
    identification_period_span: XStr = ml_model["mlIdentificationPeriod"]
    span_dates = identification_period_span.val.split(",")
    identification_period = DateRange(date.fromisoformat(span_dates[0]), date.fromisoformat(span_dates[1]))

    ml_vars = client.read_by_ids([output_var_ref] + input_var_refs)
    ml_var_point_refs = [var["mlVarPoint"] for var in ml_vars.rows if var.get("mlVarPoint")]

    return MLModel(ml_model.get("id"), ml_var_point_refs, identification_period, ml_model)


def fetch_data(client: Client, ml_model: MLModel) -> pd.DataFrame:
    data = client.his_read_by_ids(ml_model.points, ml_model.identification_period)
    return data.to_pandas().reset_index()


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df.dropna(inplace=True)
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


def update_ml_model(client: Client, ml_model: MLModel, mse: float, r2: float) -> None:
    ml_model.tags["mlModelMetrics"] = {"mse": mse, "r2": r2}
    client.commit([ml_model.tags], CommitFlag.UPDATE, False)


def write_predictions(
    client: Client,
    model: RandomForestRegressor,
    X: pd.DataFrame,
    df: pd.DataFrame,
    ml_model: MLModel,
) -> None:
    ml_prediction_point_id = None

    try:
        ml_prediction_point = client.read(f"ml and his and modelRef == @{ml_model.id.val}", 1)
        ml_prediction_point_id = ml_prediction_point.rows[0]["id"]
    except HaystackReadOpUnknownRecError:
        print("Prediction write skipped, point is missing")
        return

    df["prediction"] = model.predict(X)
    df = df[["Timestamp", "prediction"]]
    his_rows = [
        {"ts": row["Timestamp"].to_pydatetime(), "val": Number(row["prediction"])} for index, row in df.iterrows()
    ]
    client.his_write_by_ids(ml_prediction_point_id, his_rows)


def plot_results(model: RandomForestRegressor, X: pd.DataFrame, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    pred = model.predict(X)
    df["prediction"] = pred
    plt.figure(figsize=(10, 5))
    plt.plot(df["Timestamp"], df["prediction"], label="Prediction")
    plt.plot(df["Timestamp"], df[df.columns[1]], label=df.columns[1])
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main() -> None:
    env_vars = load_env_variables()
    with Client(env_vars["uri"], env_vars["username"], env_vars["password"]) as client:
        ml_model = read_ml_model(client, "mlModel and mlwgPhableDemo")
        ml_data = fetch_data(client, ml_model)
        xy, target = preprocess_data(ml_data)

        X = xy.drop(columns=[target])
        y = xy[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        mse, r2, y_pred = evaluate_model(model, X_test, y_test)
        update_ml_model(client, ml_model, mse, r2)
        write_predictions(client, model, X, ml_data, ml_model)
        plot_results(model, X, ml_data)


if __name__ == "__main__":
    main()
