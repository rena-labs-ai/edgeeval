"""MLC LLM benchmark dataset classes"""

import argparse
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # pylint: disable=import-error
from datasets import load_dataset  # pylint: disable=import-error
from transformers import AutoTokenizer  # pylint: disable=import-error

from macos_bench.request_record import GroupedRequestRecord, Metrics, RequestRecord
from macos_bench.protocol.openai_api_protocol import ChatCompletionRequest


class Dataset:  # pylint: disable=too-few-public-methods
    """The dataset base class."""

    # We set a truncation limit of 100k.
    truncate_length = int(1e5)
    # For some that datasets (e.g., dataset that has shared common prefix),
    # we need fake warmup requests to avoid prefilling common prefixes to the engine.
    require_fake_warmup: bool = False
    # Whether the dataset contains timestamps already.
    # If the dataset comes with timestamps, the benchmark can just replay
    # the requests according to their timestamps.
    timestamp_available: bool = False

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        """Get the raw unprocessed request records of the dataset."""
        raise NotImplementedError()


class ShareGPTDataset(Dataset):  # pylint: disable=too-few-public-methods
    """The dataset class for ShareGPT dataset."""

    _tokenized_dataset: List[Tuple[str, List[int], int]]
    apply_chat_template: bool

    def __init__(
        self, dataset_path: str, tokenizer: AutoTokenizer, apply_chat_template: bool
    ) -> None:
        self.apply_chat_template = apply_chat_template
        with open(dataset_path, encoding="utf-8") as f:
            raw_dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        _dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in raw_dataset
            if len(data["conversations"]) >= 2 and data["conversations"][0]["from"] == "human"
        ]
        
        self.tokenizer = tokenizer
        if tokenizer is None:
            # For Ollama, we don't need tokenization
            self._tokenized_dataset = [
                (prompt, [], len(completion.split())) for prompt, completion in _dataset
            ]
            return

        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _ in _dataset]
        if apply_chat_template:
            assert (
                getattr(tokenizer, "chat_template", None) is not None
            ), '"--apply-chat-template" is set but the tokenizer does not have chat template.'
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in prompts
            ]

        prompt_token_ids = list(
            tokenizer(
                prompts,
                truncation=True,
                max_length=min(tokenizer.model_max_length, self.truncate_length),
                add_special_tokens=False,
            ).input_ids
        )
        completions = [completion for _, completion in _dataset]
        completion_token_ids = tokenizer(
            completions,
            truncation=True,
            max_length=min(tokenizer.model_max_length, self.truncate_length),
            add_special_tokens=False,
        ).input_ids
        self._tokenized_dataset: List[Tuple[str, List[int], int]] = []
        for i in range(len(_dataset)):
            if (
                len(prompt_token_ids[i]) < 4
                or len(completion_token_ids[i]) < 4
                or len(prompt_token_ids[i]) + len(completion_token_ids[i])
                >= min(tokenizer.model_max_length, 8192)
            ):
                # Filter out sequences that are too short or too long
                continue
            self._tokenized_dataset.append(
                (prompts[i], prompt_token_ids[i], len(completion_token_ids[i]))
            )

    def generate_request_records(
        self,
        input_len: Optional[int],
        output_len: Optional[int],
        input_len_std: float = 0.0,
        output_len_std: float = 0.0,
    ) -> List[RequestRecord]:
        if self.apply_chat_template:
            assert (
                input_len is None
            ), '"--apply-chat-template" is not supported when "--input-len" is specified.'

        request_records = []
        for prompt, input_token_ids, output_length in self._tokenized_dataset:
            input_length = len(input_token_ids)
            # If the request does not have enough length, discard it.
            if input_len is not None and input_length < input_len + 4 * input_len_std:
                continue

            if input_len is not None:
                input_length = round(
                    float(np.random.normal(loc=input_len, scale=input_len_std, size=1)[0])
                )
                input_token_ids = input_token_ids[:input_length]
                input_truncated = True
            else:
                input_truncated = False
            if output_len is not None:
                output_length = round(
                    float(np.random.normal(loc=output_len, scale=output_len_std, size=1)[0])
                )
            elif output_length <= 1:
                continue
            request_records.append(
                RequestRecord(
                    chat_cmpl=ChatCompletionRequest(
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    self.tokenizer.decode(input_token_ids)
                                    if input_truncated
                                    else prompt
                                ),
                            }
                        ],
                        model="",
                        max_tokens=output_length,
                    ),
                    metrics=Metrics(
                        success=False,
                        start_time=0,
                        finish_time=0,
                        end_to_end_latency_s=0,
                        input_tokens=len(input_token_ids),
                    ),
                )
            )
        return request_records


SUPPORTED_DATASET = [
    "sharegpt",
]


def create_dataset(  # pylint: disable=too-many-return-statements,too-many-branches
    args: argparse.Namespace, tokenizer: AutoTokenizer
) -> Dataset:
    """Create a dataset instance with regard to the specified dataset kind and file path."""
    if args.dataset_path is not None and not isinstance(args.dataset_path, str):
        raise TypeError(f"Invalid dataset path {args.dataset_path}. Please use a string.")
    if args.dataset is None and args.dataset_path is not None:
        # Auto-detect the dataset kind by looking into the dataset path.
        if "sharegpt" in args.dataset_path.lower():
            args.dataset = "sharegpt"
        else:
            raise ValueError(
                f"Unable to detect the dataset kind from dataset path {args.dataset_path}. "
                'Please specify the dataset kind via "--dataset".'
            )
    if args.dataset == "sharegpt":
        if args.dataset_path is None:
            raise ValueError(
                'ShareGPT dataset requires dataset path. Please specify it with "--dataset-path".'
            )
        return ShareGPTDataset(args.dataset_path, tokenizer, args.apply_chat_template)
    raise ValueError(f"Unrecognized dataset {args.dataset}")
