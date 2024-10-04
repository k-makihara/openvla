import requests
import torch

from PIL import Image
from pathlib import Path

from prismatic import load
import json

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
#hf_token = Path(".hf_token").read_text().strip()
#hf_token = "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"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "prism-dinosiglip+7b"
vlm = load("runs/pgvlm-cvs-jpn+minimum-pgvlm-cvsjpn+stage-finetune+x7/checkpoints/step-001440-epoch-00-loss=0.0244.pt")
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
#image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"

evallist = json.load(open("/home/koshimakihara/Downloads/products/val.json", "r"))

with open("/home/koshimakihara/Downloads/products/questionnaire_vqa_max1.json", "r") as f:
    examples = json.load(f)

image_dir = "/home/koshimakihara/Downloads/products"

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
    
    user_prompt = examples[i]["question_with_options"]
    conversation = examples[i]["answer"]
    # Build prompt
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=user_prompt)
    prompt_text = prompt_builder.get_prompt()

    # Generate!1
    generated_text = vlm.generate(
        concatenated_image,
        prompt_text,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=4,
        min_length=1,
    )
    print("Eval "+str(i))
    print(generated_text)
    print(conversation)