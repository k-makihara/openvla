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
vlm = load("runs/cvs-jpn+minimum-cvsjpn+stage-finetune+x7/checkpoints/step-000290-epoch-00-loss=0.0449.pt")
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
#image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"

evallist = json.load(open("/home/koshimakihara/Downloads/products/val.json", "r"))

product_list = set([cap["name"] for cap in evallist])
user_prompt = f"What is the picture?"
#user_prompt = "What is the product name? The template of the answer is \"a picture of <product name> \""
#print(user_prompt)

for i in range(len(evallist)):
    image = Image.open("/home/koshimakihara/Downloads/products/"+evallist[i]["image_path"]).convert("RGB")

    # Build prompt
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=user_prompt)
    prompt_text = prompt_builder.get_prompt()

    # Generate!1
    generated_text = vlm.generate(
        image,
        prompt_text,
        do_sample=False,
        temperature=0.7,
        top_k=3,
        max_new_tokens=128,
        min_length=1,
    )
    print("Eval "+str(i))
    print(generated_text)
    print(evallist[i]["name"], evallist[i]["name_common"])