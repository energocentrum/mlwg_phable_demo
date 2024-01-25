import os
import os
import dotenv
from datetime import date
from phable.client import Client
from phable.kinds import DateRange, Grid,Ref,Number

dotenv.load_dotenv()
uri = os.environ["URI"]
username = os.environ["USERNAME"]
password = os.environ["PASSWORD"]

with Client(uri, username, password) as client:
  points = client.read('point and tz=="Prague" and (unit == "kW" or unit == "°C")', 2)
  grid = client.his_read(points, DateRange(date(2020, 1, 1), date(2020, 2, 1)))
  df = grid.to_pandas()
  df.dropna(inplace=True)
  dis_to_cn = {str(col_grid["meta"]["id"]): col_grid["name"] for col_grid in filter(lambda x: x["name"] != "ts", grid.cols)}
  df.rename(columns=dis_to_cn, inplace=True)

  df["prediction"] = df["v0"] + df["v1"]

  meta = {"ver": "3.0", "id": Ref("p:energyTwinForge:r:2d44cd91-c8fd065c")}
  cols = [{"name": "ts"}, {"name": "val"}]
  rows = []
  for index, row in df.iterrows():
      rows.append({"ts": index.to_pydatetime(), "val": Number(row['prediction'])})
  grid = Grid(meta, cols, rows)
  client.his_write(grid)

  #client.eval

  print(df)
  print(df.info())