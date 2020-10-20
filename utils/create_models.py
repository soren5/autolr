import os
import json
from tensorflow import keras

directory = os.path.join(os.getcwd(), "models", "json")


for filename in os.scandir(directory):
    path = os.path.join(directory, filename)
    print(path)
    with open(path, "r") as f:
        model_json = json.load(f)
        model = keras.models.model_from_json(model_json)
        path = os.path.join(os.getcwd(), "models", filename.name[:-5] + ".h5")
        model.save(path)