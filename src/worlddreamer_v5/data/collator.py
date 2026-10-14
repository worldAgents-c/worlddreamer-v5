# Copyright (C) 2026 World Agents Corporation.
import os
from typing import Dict
import torch
from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor
from worlddreamer_v5.data.schema import SCORE_ID, STATE_ID, IMAGE_ID


class WorldCollator:
    def __init__(self, processor_path) -> None:
        self.max_length = int(os.environ.get("MAX_LENGTH", 20000))
        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(60)})
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            local_files_only=True,
            use_fast=True,
            extra_special_tokens=special_tokens,
        )
        token_ids = self.processor.tokenizer.convert_tokens_to_ids(
            ["<score>", "<state>", "<a_0>", "<a_59>"]
        )
        if token_ids != [SCORE_ID, STATE_ID, STATE_ID + 1, STATE_ID + 60]:
            raise ValueError(f"unexpected action token ids: {token_ids}")

    def _position_ids(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        grids = inputs.get("image_grid_thw")
        if grids is None:
            return torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)

        tokens = input_ids[0].tolist()
        positions, start = [], 0
        for grid in grids:
            image_start = tokens.index(IMAGE_ID, start)
            text_length = image_start - start
            base = positions[-1].max() + 1 if positions else 0
            positions.append(torch.arange(text_length).view(1, -1).expand(3, -1) + base)
            time, height, width = (int(value) for value in grid)
            height //= self.processor.image_processor.merge_size
            width //= self.processor.image_processor.merge_size
            temporal = (
                torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            )
            rows = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            columns = (
                torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            )
            positions.append(
                torch.stack([temporal, rows, columns]) + text_length + base
            )
            start = image_start + time * height * width

        if start < len(tokens):
            base = positions[-1].max() + 1
            positions.append(
                torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + base
            )
        return torch.cat(positions, dim=1).view(3, 1, -1)

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError(
                f"worlddreamer-v5 requires per-device scene batch size 1, got {len(batch)}"
            )

        item = batch[0]
        inputs = self.processor.apply_chat_template(
            item["messages"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
        )
        sequence_length = int(inputs["input_ids"].shape[1])
        if sequence_length > self.max_length:
            raise ValueError(
                f"scene length {sequence_length} exceeds MAX_LENGTH={self.max_length}"
            )

        query_positions = (
            (inputs["input_ids"][0] == STATE_ID).nonzero(as_tuple=False).flatten()
        )
        expected_queries = int(item["agent_mask"].sum().item())
        if query_positions.numel() != expected_queries:
            raise ValueError(
                f"found {query_positions.numel()} VLM queries, expected {expected_queries}"
            )

        inputs["position_ids"] = self._position_ids(inputs)
        result = {
            "input_ids": inputs["input_ids"],
            "position_ids": inputs["position_ids"],
            "scene_length": torch.tensor([int(query_positions[0])], dtype=torch.long),
        }
        if inputs.get("attention_mask") is not None:
            result["attention_mask"] = inputs["attention_mask"]
        for key in ("pixel_values", "image_grid_thw"):
            if inputs.get(key) is not None:
                result[key] = inputs[key]

        payload = {key: value for key, value in item.items() if key != "messages"}
        result.update(default_collate([payload]))
        return result
