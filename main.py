import os
import os
import dotenv
from datetime import date
from phable.client import Client, CommitFlag
from phable.kinds import DateRange, Grid, Ref, Number, Marker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dotenv.load_dotenv()
uri = os.environ["URI"]
username = os.environ["USERNAME"]
password = os.environ["PASSWORD"]

with Client(uri, username, password) as client:

    ml_models = client.read(
        'mlModel and dis=="ft - 15min"'
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

    X = df[[df.columns[0]]]
    y = df[df.columns[1:]]
    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)

    # Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label="Measured", color="b")
    plt.plot(range(len(y_test)), y_pred, label="Predicted", color="r", linestyle="--")
    plt.xlabel("Index")
    plt.ylabel("Class")
    plt.title("Measured vs. Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # data = [{"dis": "TestRec", "testing": Marker(), "pytest": Marker()}]
    # response: Grid = client.commit(data, CommitFlag.ADD, False)

    # points = client.read('point and tz=="Prague" and (unit == "kW" or unit == "°C")', 2)
    # grid = client.his_read(points, DateRange(date(2020, 1, 1), date(2020, 2, 1)))
    # df = grid.to_pandas()
    # df.dropna(inplace=True)
    # dis_to_cn = {
    #    str(col_grid["meta"]["id"]): col_grid["name"]
    #    for col_grid in filter(lambda x: x["name"] != "ts", grid.cols)
    # }
    # df.rename(columns=dis_to_cn, inplace=True)
    #
    # df["prediction"] = df["v0"] + df["v1"]
    #
    # meta = {"ver": "3.0", "id": Ref("p:energyTwinForge:r:2d44cd91-c8fd065c")}
    # cols = [{"name": "ts"}, {"name": "val"}]
    # rows = []
    # for index, row in df.iterrows():
    #    rows.append({"ts": index.to_pydatetime(), "val": Number(row["prediction"])})
    # grid = Grid(meta, cols, rows)
    # client.his_write(grid)
    #
    ## client.eval
    #
    # print(df)
    # print(df.info())
