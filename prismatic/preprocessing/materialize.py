"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""

from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import AlignDataset, FinetuneDataset, CVSFinetuneDataset, PGVLM_CVSFinetuneDataset, PGVLM_CVSAlignDataset, PGVLM_CVS_BTFinetuneDataset, PGVLM_CVS_PrefAlignDataset, PGVLM_CVS_PrefFinetuneDataset
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling, PaddedCollatorForPGLanguageModeling, PaddedCollatorForPreferencePrediction

from prismatic.preprocessing.preference_tokenizer import PreferenceTokenizer

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": PGVLM_CVSAlignDataset, "finetune": PGVLM_CVSFinetuneDataset, "full-finetune": PGVLM_CVSFinetuneDataset, "last-layer-finetune": PGVLM_CVSFinetuneDataset}
DATASET_INITIALIZER_PREF = {"align": PGVLM_CVS_PrefAlignDataset, "finetune": PGVLM_CVS_PrefFinetuneDataset, "full-finetune": PGVLM_CVS_PrefFinetuneDataset, "last-layer-finetune": PGVLM_CVS_PrefFinetuneDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator
    
    elif stage == "last-layer-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")


def get_pref_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PreferenceTokenizer, PaddedCollatorForPreferencePrediction]:
    dataset_cls = DATASET_INITIALIZER_PREF[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    preference_tokenizer = PreferenceTokenizer(tokenizer)
    collator = PaddedCollatorForPreferencePrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer, preference_tokenizer
        )
        return dataset, preference_tokenizer, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            preference_tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        #print(dataset[0])
        return dataset, preference_tokenizer, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            preference_tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, preference_tokenizer, collator
    
    elif stage == "last-layer-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            preference_tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, preference_tokenizer, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
