# About

This demo connects to haystack server, reads ml model, downloads ml var data, trains random forest model, calculates metrics which are saved back to ml model on server.

After that it calculates prediction which is showed in pyplot chart, and also saved on server in ml prediction point.

# Prerequisites
- Clone this repository
  - Install dependencies `pip install -r ./requirements.txt`
- Connection to haystack server like [haxall](https://haxall.io/) or [skyspark](https://skyfoundry.com/product)
  - Create `.env` based on `.env.template` with your credentials
- Create ml model with all required tags
  - At least one input and one output var
  - Models are filtered using extra tag `mlwgPhableDemo` on ml model
    - You can modify this filter in `main` function in `main.py`
- Run demo `python ./main.py`

### Optional
- Create point which will be used to store calculated prediction
  - With mlwg tags `ml`, `his` and `modelRef` referencing ml model