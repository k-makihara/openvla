import requests
import json_numpy
json_numpy.patch()
import numpy as np
rng = np.random.default_rng()

for i in range(10):
    img = rng.integers(0, high=256, size=(256, 256, 3), dtype=np.uint8, endpoint=False)

    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": img, "instruction": "do something", "unnorm_key": "bc_z"}
    ).json()
    print(action)