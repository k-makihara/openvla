import requests
import torch

from PIL import Image
from pathlib import Path

from prismatic import load
import json

import matplotlib.pyplot as plt
import numpy as np
import random

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
#hf_token = Path(".hf_token").read_text().strip()
#hf_token = "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"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
#model_id = "prism-dinosiglip+7b"
vlm = load("/home/koshimakihara/openvla/runs/pgvlm-cvs-jpn+minimum-pgvlm-cvsjpn-pref-v4+stage-finetune+x7/checkpoints/step-046069-epoch-00-loss=0.0000.pt")
#vlm = load("/home/koshimakihara/openvla/runs/pgvlm-cvs-jpn+minimum-pgvlm-cvsjpn-v5+stage-finetune+x7/checkpoints/step-046069-epoch-00-loss=0.1173.pt")
#vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

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
        
        user_prompt = examples[i]["question_with_options"][1:-1]
        conversation = examples[i]["answer"]

        #plt.figure()
        #plt.title("Question: " + user_prompt + " Answer: " + conversation)
        #plt.imshow(np.array(concatenated_image))
        #plt.show()
        # Build prompt
        #print("Question: " + user_prompt + " Answer: " + conversation)
        #concatenated_image.show()
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()
        #print(prompt_text)
        # Generate!1
        #generated_text = vlm.generate(
        #    concatenated_image,
        #    prompt_text,
        #    do_sample=True,
        #    temperature=0.4,
        #    max_new_tokens=512,
        #    min_length=1,
        #)
        #generated_texts, scores = vlm.generate_score(
        #    concatenated_image,
        #    prompt_text,
        #    do_sample=False,
        #    #temperature=0.2,
        #    return_dict_in_generate=True, output_scores=True,num_return_sequences=3,
        #    num_beams=3,
        #    max_new_tokens=2,
        #    min_length=1,
        #    top_p=0.9,
        #    top_k=50,
        #    repetition_penalty=1.5,
        #    length_penalty=0,
        #)
        generated_preference = vlm.generate_preference(
            concatenated_image,
            prompt_text,
            do_sample=False,
            #temperature=0.7,
            max_new_tokens=2,
            #min_length=1,
        )
        print("Eval "+str(i+1))
        #print(generated_text)
        #print(generated_texts, scores)
        #print(generated_preference)
        if generated_preference[0] > generated_preference[1]:
            ans = "left"
        else:
            ans = "right"
        if abs(generated_preference[0] - generated_preference[1]) < 0.02:
            ans = "tie"
        print(ans, conversation)

        #if conversation == ans:
        #    accuracy = accuracy + 1

    print(accuracy / len(examples) * 100)