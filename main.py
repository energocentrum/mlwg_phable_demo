import os
import os
import dotenv
from datetime import date
from phable.client import Client, CommitFlag
from phable.kinds import DateRange, Grid, Ref, Number, Marker
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dotenv.load_dotenv()
uri = os.environ["URI"]
username = os.environ["USERNAME"]
password = os.environ["PASSWORD"]

with Client(uri, username, password) as client:

    ml_models = client.read(
        'mlModel and dis=="mlwg model"'
    )  # ml_models = client.read("mlModel")

    ml_model = ml_models.rows[0]  # for ml_model in ml_models.rows:

    # get refs of all vars
    ml_output_ref: Ref = ml_model.get("mlOutputVarRef", None)
    ml_input_refs: list[dict] = ml_model.get("mlInputVarRefs", None)
    ml_var_refs: list[Ref] = [ml_output_ref] + [
        Ref(ref["val"], ref["dis"]) for ref in ml_input_refs
    ]
    # read var records
    ml_vars = client.read_by_ids(ml_var_refs)

    # extract point refs and read data
    ml_var_points = [
        var["mlVarPoint"] for var in ml_vars.rows if var.get("mlVarPoint") != None
    ]
    data = client.his_read_by_ids(
        ml_var_points, DateRange(date(2023, 1, 1), date(2023, 2, 1))
    )

    df = data.to_pandas()
    df = df.reset_index()
    print(df)

    target = df.columns[1]
    print(target)

    # Extract useful features from 'Timestamp'
    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.day
    df["month"] = df["Timestamp"].dt.month
    df["dayofweek"] = df["Timestamp"].dt.dayofweek

    # Drop the original 'Timestamp' column
    xy = df.drop(columns=["Timestamp"])

    # 4. Feature Selection/Engineering
    # Define features (X) and target (y)
    X = xy.drop(columns=[target])
    y = xy[target]

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train the Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    params = model.get_params()
    params = [p for p in params if p != None]

    ml_model["mlModelMetrics"] = {"mse": mse, "r2": r2}
    ml_model["mlModelParameters"] = params
    response: Grid = client.commit([ml_model], CommitFlag.UPDATE, False)

    pred = model.predict(X)
    df["prediction"] = pred
    df = df[["Timestamp", "prediction"]]

    ml_prediction_point = client.read(
        f'mlPrediction and his and mlModelRef == @{ml_model["id"].val}'
    )

    his_rows = []
    for index, row in df.iterrows():
        his_rows.append(
            {"ts": row["Timestamp"].to_pydatetime(), "v0": Number(row["prediction"])}
        )
    client.his_write_by_ids([ml_prediction_point.rows[0]["id"]], his_rows)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()
