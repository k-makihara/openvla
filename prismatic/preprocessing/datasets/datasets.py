"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type
import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CodeGenTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase
from prismatic.preprocessing.preference_tokenizer import PreferenceTokenizer

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations"]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()

            # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
            elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                pass

            elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
                pass

            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
            #print(pixel_values)
            #print(input_ids)
            #print(labels)
            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class CVSFinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        product_name = self.examples[idx]["name"]
        common_name = self.examples[idx]["name_common"]
        user_prompt = f"What is the picture?"
        conversation = f"a picture of {product_name} which is commonly {common_name}"

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []

        turn_idx = 0
        # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
        msg = prompt_builder.add_turn("human", user_prompt)
        print(msg)

        # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            msg = msg.rstrip()

        # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
        elif isinstance(self.tokenizer, CodeGenTokenizerFast):
            pass

        elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pass

        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

        # Tokenize Input IDs
        turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
        #print(turn_input_ids)

        # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
        turn_labels = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
        )
        #turn_labels = (
        #    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if False else list(turn_input_ids)
        #)
        #print(turn_labels)
        # Add to Trackers
        input_ids.extend(turn_input_ids)
        labels.extend(turn_labels)

        turn_idx = 1
        # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
        msg = prompt_builder.add_turn("gpt", conversation)
        print(msg)

        # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            msg = msg.rstrip()

        # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
        elif isinstance(self.tokenizer, CodeGenTokenizerFast):
            pass

        elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pass

        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

        # Tokenize Input IDs
        turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
        #print(turn_input_ids)

        # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
        turn_labels = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
        )
        #turn_labels = (
        #    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if False else list(turn_input_ids)
        #)
        #print(turn_labels)
        # Add to Trackers
        input_ids.extend(turn_input_ids)
        labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image_path" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image_path"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
            #print(pixel_values)
            #print(input_ids)
            #print(labels)
            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image_path" in example
            n_words = len(example["name"].split()) + len(example["name_common"].split()) + 6
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class PGVLM_CVSAlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        #self.prompt_builder_fn = prompt_builder_fn

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        self.dataset_type = "align"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        user_prompt = self.examples[idx]["question_with_options"]
        conversation = self.examples[idx]["answer"]
        
        all_conversation = "Question: " + user_prompt + "Short Answer: " + conversation
        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=all_conversation.strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])

        image_path_left = Path(self.examples[idx]["image_paths"]["left_image"])
        image_path_right = Path(self.examples[idx]["image_paths"]["right_image"])

        # 画像を開く
        left_image = Image.open(self.image_dir / image_path_left).convert("RGB")
        right_image = Image.open(self.image_dir / image_path_right).convert("RGB")

        # キャンバスのサイズを計算（横幅は画像の合計幅、高さは一番高い画像の高さ）
        total_width = left_image.width + right_image.width
        max_height = max(left_image.height, right_image.height)

        # 新しい画像（キャンバス）を作成
        concatenated_image = Image.new("RGB", (total_width, max_height))

        # 左画像を貼り付け
        concatenated_image.paste(left_image, (0, 0))

        # 右画像を左画像の横に貼り付け
        concatenated_image.paste(right_image, (left_image.width, 0))

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(concatenated_image)

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image_paths" in example
            cap = "Question: " + example["question_with_options"] + "Short Answer: " + cexample["answer"]
            n_words = len(cap.split())
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class PGVLM_CVSFinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        flip = random.random()

        user_prompt = self.examples[idx]["question_with_options"][1:-1]
        conversation = self.examples[idx]["answer"]
        if flip > 0.5:
            if conversation == "left":
                conversation = "right"
            elif conversation == "right":
                conversation = "left"

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []

        turn_idx = 0
        # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
        msg = prompt_builder.add_turn("human", user_prompt)
        #print(msg)

        # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            msg = msg.rstrip()

        # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
        elif isinstance(self.tokenizer, CodeGenTokenizerFast):
            pass

        elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pass

        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

        # Tokenize Input IDs
        #print(self.tokenizer)
        #print(msg)
        #turn_input_ids = self.tokenizer(msg, add_special_tokens=True).input_ids
        turn_input_ids = self.tokenizer(msg, add_special_tokens=False).input_ids
        #print(turn_input_ids)

        # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
        turn_labels = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
        )
        #turn_labels = (
        #    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if False else list(turn_input_ids)
        #)
        #print(turn_labels)
        # Add to Trackers
        input_ids.extend(turn_input_ids)
        labels.extend(turn_labels)

        turn_idx = 1
        # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
        msg = prompt_builder.add_turn("gpt", conversation)
        #print(msg)

        # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            msg = msg.rstrip()

        # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
        elif isinstance(self.tokenizer, CodeGenTokenizerFast):
            pass

        elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pass

        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

        # Tokenize Input IDs
        #print(msg)
        #turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
        turn_input_ids = self.tokenizer(msg, add_special_tokens=False).input_ids
        #print(turn_input_ids)

        # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
        turn_labels = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
        )
        #turn_labels = (
        #    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if False else list(turn_input_ids)
        #)
        #print(turn_labels)
        # Add to Trackers
        input_ids.extend(turn_input_ids)
        labels.extend(turn_labels)
        #print(input_ids)
        #print(labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image_paths" in self.examples[idx]:
            if flip > 0.5:
                image_path_left = Path(self.examples[idx]["image_paths"]["right_image"])
                image_path_right = Path(self.examples[idx]["image_paths"]["left_image"])
            else:
                image_path_left = Path(self.examples[idx]["image_paths"]["left_image"])
                image_path_right = Path(self.examples[idx]["image_paths"]["right_image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX
            #labels[:] = IGNORE_INDEX

            # 画像を開く
            left_image = Image.open(self.image_dir / image_path_left).convert("RGB")
            right_image = Image.open(self.image_dir / image_path_right).convert("RGB")

            # キャンバスのサイズを計算（横幅は画像の合計幅、高さは一番高い画像の高さ）
            total_width = left_image.width + right_image.width
            max_height = max(left_image.height, right_image.height)

            # 新しい画像（キャンバス）を作成
            concatenated_image = Image.new("RGB", (total_width, max_height))

            # 左画像を貼り付け
            concatenated_image.paste(left_image, (0, 0))

            # 右画像を左画像の横に貼り付け
            concatenated_image.paste(right_image, (left_image.width, 0))

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(concatenated_image)
            #print(pixel_values)
            #print(input_ids)
            #print(labels)
            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            labels[0] = IGNORE_INDEX
            #labels[-3:] = IGNORE_INDEX
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image_paths" in example
            n_words = len(example["question_with_options"].split())
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class PGVLM_CVS_BTFinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        user_prompt = self.examples[idx]["question_with_options"]
        conversation = self.examples[idx]["answer"]

        answer_preference = np.asarray(np.zeros(2), dtype=np.float32)
        if conversation == "left":
            answer_preference = np.asarray([1.0, 0.0], dtype=np.float32)
        elif conversation == "right":
            answer_preference = np.asarray([0.0, 1.0], dtype=np.float32)
        elif conversation == "tie":
            answer_preference = np.asarray([0.5, 0.5], dtype=np.float32)


        # Create Prompt Builder --> add each message sequentially
        prompt_builder_yes, prompt_builder_no, input_ids_yes, input_ids_no, labels_yes, labels_no = self.prompt_builder_fn(model_family="prismatic"), self.prompt_builder_fn(model_family="prismatic"), [], [], [], []

        turn_idx = 0
        # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
        msg_yes = prompt_builder_yes.add_turn("human", user_prompt)
        msg_no = prompt_builder_no.add_turn("human", user_prompt)
        #print(msg)

        # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            msg_yes = msg_yes.rstrip()
            msg_no = msg_no.rstrip()

        # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
        elif isinstance(self.tokenizer, CodeGenTokenizerFast):
            pass

        elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pass

        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

        # Tokenize Input IDs
        turn_input_ids_yes = self.tokenizer(msg_yes, add_special_tokens=turn_idx == 0).input_ids
        turn_input_ids_no = self.tokenizer(msg_no, add_special_tokens=turn_idx == 0).input_ids
        #print(turn_input_ids)

        # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
        turn_labels_yes = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids_yes))] if (turn_idx % 2) == 0 else list(turn_input_ids_yes)
        )
        turn_labels_no = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids_no))] if (turn_idx % 2) == 0 else list(turn_input_ids_no)
        )
        #turn_labels = (
        #    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if False else list(turn_input_ids)
        #)
        #print(turn_labels)
        # Add to Trackers
        input_ids_yes.extend(turn_input_ids_yes)
        input_ids_no.extend(turn_input_ids_no)
        labels_yes.extend(turn_labels_yes)
        labels_no.extend(turn_labels_no)

        turn_idx = 1
        # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
        msg_yes = prompt_builder_yes.add_turn("gpt", "yes")
        msg_no = prompt_builder_no.add_turn("gpt", "no")
        #print(msg)

        # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            msg_yes = msg_yes.rstrip()
            msg_no = msg_no.rstrip()

        # Phi-2 Tokenizer == CodeGenTokenizer (Fast) -- no special handling!
        elif isinstance(self.tokenizer, CodeGenTokenizerFast):
            pass

        elif isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pass

        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

        # Tokenize Input IDs
        turn_input_ids_yes = self.tokenizer(msg_yes, add_special_tokens=turn_idx == 0).input_ids
        turn_input_ids_no = self.tokenizer(msg_no, add_special_tokens=turn_idx == 0).input_ids
        #print(turn_input_ids)

        # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
        turn_labels_yes = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids_yes))] if (turn_idx % 2) == 0 else list(turn_input_ids_yes)
        )
        turn_labels_no = (
            [IGNORE_INDEX for _ in range(len(turn_input_ids_no))] if (turn_idx % 2) == 0 else list(turn_input_ids_no)
        )
        #turn_labels = (
        #    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if False else list(turn_input_ids)
        #)
        #print(turn_labels)
        # Add to Trackers
        input_ids_yes.extend(turn_input_ids_yes)
        labels_yes.extend(turn_labels_yes)
        input_ids_no.extend(turn_input_ids_no)
        labels_no.extend(turn_labels_no)

        


        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids_yes, labels_yes, input_ids_no, labels_no = torch.tensor(input_ids_yes), torch.tensor(labels_yes), torch.tensor(input_ids_no), torch.tensor(labels_no)

        # Handle Truncation (if necessary)
        input_ids_yes, labels_yes, input_ids_no, labels_no = input_ids_yes[: self.tokenizer.model_max_length], labels_yes[: self.tokenizer.model_max_length], input_ids_no[: self.tokenizer.model_max_length], labels_no[: self.tokenizer.model_max_length]

        pixel_values = []

        input_ids = []
        labels = []
        input_ids.append(input_ids_yes)
        input_ids.append(input_ids_no)
        labels.append(labels_yes)
        labels.append(labels_no)
        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image_paths" in self.examples[idx]:
            image_path_left = Path(self.examples[idx]["image_paths"]["left_image"])
            image_path_right = Path(self.examples[idx]["image_paths"]["right_image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels_yes[0] = IGNORE_INDEX
            labels_no[0] = IGNORE_INDEX

            # 画像を開く
            left_image = Image.open(self.image_dir / image_path_left).convert("RGB")
            right_image = Image.open(self.image_dir / image_path_right).convert("RGB")

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values.append(self.image_transform(left_image))
            pixel_values.append(self.image_transform(right_image))
            #print(pixel_values)
            #print(input_ids)
            #print(labels)
            return dict(pixel_values_left=self.image_transform(left_image), pixel_values_right=self.image_transform(right_image), input_ids_yes=input_ids_yes, input_ids_no=input_ids_no, labels_yes=labels_yes, labels_no=labels_no, labels_value=torch.from_numpy(answer_preference))

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels, labels_value=torch.from_numpy(answer_preference))

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image_paths" in example
            n_words = len(example["question_with_options"].split())
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

class PGVLM_CVS_PrefAlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        preference_tokenizer: PreferenceTokenizer,
        predict_stop_token: bool = True
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer, self.preference_tokenizer = image_transform, tokenizer, preference_tokenizer
        #self.prompt_builder_fn = prompt_builder_fn

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        self.dataset_type = "align"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        user_prompt_ori = self.examples[idx]["question_with_options"]
        if "deformable" in user_prompt_ori:
            user_prompt = "Is this object deformable?"
        elif "vulnerable" in user_prompt_ori:
            user_prompt = "Is this object vulnerable?"
        elif "slippery" in user_prompt_ori:
            user_prompt = "Is this object slippery?"
        else:
            user_prompt = "Is this object better?"

        conversation = self.examples[idx]["answer"]
        
        

        caption = self.prompt_template.format(caption=all_conversation.strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])

        image_path_left = Path(self.examples[idx]["image_paths"]["left_image"])
        image_path_right = Path(self.examples[idx]["image_paths"]["right_image"])

        # 画像を開く
        left_image = Image.open(self.image_dir / image_path_left).convert("RGB")
        right_image = Image.open(self.image_dir / image_path_right).convert("RGB")

        # キャンバスのサイズを計算（横幅は画像の合計幅、高さは一番高い画像の高さ）
        total_width = left_image.width + right_image.width
        max_height = max(left_image.height, right_image.height)

        # 新しい画像（キャンバス）を作成
        concatenated_image = Image.new("RGB", (total_width, max_height))

        # 左画像を貼り付け
        concatenated_image.paste(left_image, (0, 0))

        # 右画像を左画像の横に貼り付け
        concatenated_image.paste(right_image, (left_image.width, 0))

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(concatenated_image)

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image_paths" in example
            cap = "Question: " + example["question_with_options"] + "Short Answer: " + example["answer"]
            n_words = len(cap.split())
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class PGVLM_CVS_PrefFinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        preference_tokenizer: PreferenceTokenizer,
        prompt_builder_fn: Type[PromptBuilder],
        predict_stop_token: bool = True
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer, self.preference_tokenizer = image_transform, tokenizer, preference_tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        flip = random.random()

        user_prompt = self.examples[idx]["question_with_options"][1:-1]
        conversation = self.examples[idx]["answer"]

        if flip > 0.5:
            if conversation == "left":
                conversation = "right"
            elif conversation == "right":
                conversation = "left"

        answer_preference = np.asarray(np.zeros(2), dtype=np.float32)
        if conversation == "left":
            answer_preference = np.asarray([1.0, 0.0], dtype=np.float32)
        elif conversation == "right":
            answer_preference = np.asarray([0.0, 1.0], dtype=np.float32)
        elif conversation == "tie":
            answer_preference = np.asarray([0.5, 0.5], dtype=np.float32)

        prompt_builder = self.prompt_builder_fn("prismatic")
        conversation = [
            {"from": "human", "value": f"{user_prompt}"},
            {"from": "gpt", "value": self.preference_tokenizer(answer_preference)},
        ]
        #print("gg")
        #print(self.preference_tokenizer(answer_preference))
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        #print("GGG")
        # Tokenize (w/ `base_tokenizer`)
        #print(prompt_builder.get_prompt())
        #prompt = prompt_builder.get_prompt().rstrip()
        input_ids = self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        #generated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()
        #print(input_ids)
        #print(self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True))

        
        #print(generated_text)
        #pref_input_ids = self.tokenizer("[/INST]\n" + self.preference_tokenizer(answer_preference)+"</s>", add_special_tokens=True).input_ids
        #generated_text = self.tokenizer.decode(pref_input_ids, skip_special_tokens=True).strip()
        #print(pref_input_ids)
        #print(generated_text)
        labels = list(input_ids)
        
        
        #print(labels[-10:])

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image_paths" in self.examples[idx]:
            if flip > 0.5:
                image_path_left = Path(self.examples[idx]["image_paths"]["right_image"])
                image_path_right = Path(self.examples[idx]["image_paths"]["left_image"])
            else:
                image_path_left = Path(self.examples[idx]["image_paths"]["left_image"])
                image_path_right = Path(self.examples[idx]["image_paths"]["right_image"])

            # 画像を開く
            left_image = Image.open(self.image_dir / image_path_left).convert("RGB")
            right_image = Image.open(self.image_dir / image_path_right).convert("RGB")

            # キャンバスのサイズを計算（横幅は画像の合計幅、高さは一番高い画像の高さ）
            total_width = left_image.width + right_image.width
            max_height = max(left_image.height, right_image.height)

            # 新しい画像（キャンバス）を作成
            concatenated_image = Image.new("RGB", (total_width, max_height))

            # 左画像を貼り付け
            concatenated_image.paste(left_image, (0, 0))

            # 右画像を左画像の横に貼り付け
            concatenated_image.paste(right_image, (left_image.width, 0))

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(concatenated_image)
            
            #print(len(labels))
            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(len(answer_preference)+2)] = IGNORE_INDEX
            #labels[-2:] = IGNORE_INDEX
            #labels[-3:] = IGNORE_INDEX
            #print(labels)
            #if not self.predict_stop_token:
            #    labels[-1] = IGNORE_INDEX

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            labels[: -(len(answer_preference)+2)] = IGNORE_INDEX
            #labels[-2:] = IGNORE_INDEX
            #labels[-3:] = IGNORE_INDEX
            #if not self.predict_stop_token:
            #    labels[-1] = IGNORE_INDEX

            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image_paths" in example
            n_words = len(example["question_with_options"].split())
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)

