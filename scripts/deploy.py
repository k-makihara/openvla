"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path
import os

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic import load

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
hf_token = "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"

def get_pgvlm_prompt(instruction: str, pgvlm_path: Union[str, Path]) -> str:
    if "v01" in pgvlm_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class PGVLMServer:
    def __init__(self, pgvlm_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.pgvlm_path, self.attn_implementation = pgvlm_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        #self.processor = AutoProcessor.from_pretrained(self.pgvlm_path, trust_remote_code=True)
        #self.vla = AutoModelForVision2Seq.from_pretrained(
        #    self.pgvlm_path,
        #    attn_implementation=attn_implementation,
        #    torch_dtype=torch.bfloat16,
        #    low_cpu_mem_usage=True,
        #    trust_remote_code=True,
        #).to(self.device)

        if os.path.isfile(self.pgvlm_path):
            self.vlm = load(self.pgvlm_path)
        else:
            self.vlm = load(self.pgvlm_path, hf_token=hf_token)
        self.vlm.to(self.device, dtype=torch.bfloat16)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        #if os.path.isdir(self.pgvlm_path):
        #    with open(Path(self.pgvlm_path) / "dataset_statistics.json", "r") as f:
        #        self.vla.norm_stats = json.load(f)

    def predict_preference(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]

            # Run VLA Inference
            #inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)

            im = Image.fromarray(image).convert("RGB")

            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=instruction)
            prompt_text = prompt_builder.get_prompt()
            #print(prompt_text)
            # Generate!1
            generated_text = self.vlm.generate(
                im,
                prompt_text,
                do_sample=False,
                #temperature=0.4,
                max_new_tokens=512,
                min_length=1,
            )
            #generated_texts, scores = self.vlm.generate_score(
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
            #generated_preference = self.vlm.generate_preference(
            #    im,
            #    prompt_text,
            #    do_sample=False,
            #    #temperature=0.7,
            #    max_new_tokens=2,
            #    #min_length=1,
            #)

            #action = self.vla.predict_action(**inputs, do_sample=False)
            #print(action)
            if double_encode:
                return JSONResponse(json_numpy.dumps(generated_text))
            else:
                return JSONResponse(generated_text)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_preference)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    #pgvlm_path: Union[str, Path] = "/home/koshimakihara/openvla/runs/pgvlm-cvs-jpn+minimum-pgvlm-cvsjpn-pref-v4+stage-finetune+x7/checkpoints/step-046069-epoch-00-loss=0.0543.pt"               # HF Hub Path (or path to local run directory)
    pgvlm_path: Union[str, Path] = "prism-dinosiglip+7b"
    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = PGVLMServer(cfg.pgvlm_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
