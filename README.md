# About

This demo connects to haystack server, reads ml model, downloads ml var data, trains random forest model, calculates metrics which are saved back to ml model on server.

After that it calculates prediction which is showed in pyplot chart, and also saved on server in ml prediction point.

# Prerequisites

- Clone this repository
  - Install dependencies `pip install -r ./requirements.txt`
- Connection to haystack server
  - Create `.env` based on `.env.template` with your credentials
- Make sure you have `mlModel` created on the server
  - If you dont have one, you can follow [Quick start example](#Quick-start-example)

### Optional

- Create point which will be used to store calculated prediction
  - With mlwg tags `ml`, `his` and `modelRef` referencing ml model
- This demo will look for `mlModel` with extra tag `mlwgPhableDemo`
  - You can modify this filter in `main` function in `main.py`

# Running demo

Run demo using `python ./main.py`.

# Quick start example

You may need to update the paths to example files and/or modify the import scripts to work on your haystack server.

- Create mlModel and other required example records
  - ```ioReadTrio(`io/example_records.trio`).each(x=>diff(null,x,{add}).commit)```
- Write input var data
  - ```ioReadZinc(`io/example_input_var_data.zinc`).hisWrite(readById(@2d269048-0d827cf9))```
- Write output var data
  - ```ioReadZinc(`io/example_output_var_data.zinc`).hisWrite(readById(@2e2cd829-bea309a9))```
- Complete prerequisites
  - see [Prerequisites](#Prerequisites)
- Run demo
  - see [Running demo](#Running-demo)
- Compare prediction
  - ```readByIds([@2df5a426-704c4ace,@2e2cd829-bea309a9,@2d269048-0d827cf9]).hisRead(2023-01)```
