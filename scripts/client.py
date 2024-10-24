import requests
import json_numpy
json_numpy.patch()
import numpy as np
import json
import random
from PIL import Image
from pathlib import Path
rng = np.random.default_rng()

for i in range(10):
    img = rng.integers(0, high=256, size=(256, 256, 3), dtype=np.uint8, endpoint=False)#

    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": img, "instruction": "do something", "unnorm_key": "bc_z"}
    ).json()
    print(action)

"""
rand_num = 10
for j in range(rand_num):
    with open("/home/koshimakihara/Downloads/products/questionnaire_vqa_max2_v1_3.json", "r") as f:
        examples = json.load(f)
        random.seed(j)
        random.shuffle(examples)
        examples = examples[:100]



    image_dir = "/home/koshimakihara/Downloads/products"

    accuracy = 0
    for i in range(len(examples)):
        image_path_left = Path(examples[i]["image_paths"]["left_image"])
        image_path_right = Path(examples[i]["image_paths"]["right_image"])

        # 画像を開く
        left_image = Image.open(image_dir / image_path_left).convert("RGB")
        right_image = Image.open(image_dir / image_path_right).convert("RGB")

        # キャンバスのサイズを計算（横幅は画像の合計幅、高さは一番高い画像の高さ）
        total_width = left_image.width + right_image.width
        max_height = max(left_image.height, right_image.height)

        # 新しい画像（キャンバス）を作成
        concatenated_image = Image.new("RGB", (total_width, max_height))

        # 左画像を貼り付け
        concatenated_image.paste(left_image, (0, 0))

        # 右画像を左画像の横に貼り付け
        concatenated_image.paste(right_image, (left_image.width, 0))
        image_np = np.array(concatenated_image, dtype=np.uint8)
        
        user_prompt = examples[i]["question_with_options"][1:-1]
        conversation = examples[i]["answer"]

        generated_text = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": image_np, "instruction": conversation, "unnorm_key": "bc_z"}
        ).json()

        print("Eval "+str(i+1))
        print(generated_text)
        #print(generated_texts, scores)
        #print(generated_preference)
        #if generated_preference[0] > generated_preference[1]:
        #    ans = "left"
        #else:
        #    ans = "right"
        #if abs(generated_preference[0] - generated_preference[1]) < 0.02:
        #    ans = "tie"
        #print(ans, conversation)

        #if conversation == ans:
        #    accuracy = accuracy + 1

    print(accuracy / len(examples) * 100)
"""