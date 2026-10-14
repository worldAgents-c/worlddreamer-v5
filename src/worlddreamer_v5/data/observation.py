# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import random
import re

import numpy as np
from PIL import Image

from worlddreamer_v5.utils.normalization import (
    compose_state,
    get_value,
    resize_image,
    rotm2aa_batch,
)

PROMPT_RE = re.compile(r"(<image>|<video>)")


class ObservationTools:
    @staticmethod
    def _images(traj, keys, frame):
        from decord import VideoReader

        images = []
        frame = [max(frame, 0)]
        for key in keys:
            infos = get_value(traj, key)
            if infos is None:
                raise ValueError(
                    f"{key} is missing in trajectory {traj.get('time', '<unknown>')}"
                )
            for info in infos:
                video = VideoReader(info["path"], num_threads=2)
                video.seek(0)
                images.extend(
                    Image.fromarray(image)
                    for image in ObservationTools._crop_bbox(
                        video.get_batch(
                            [frame[0] + int(info.get("start", 0))]
                        ).asnumpy(),
                        info.get("crop_bbox"),
                    )
                )
        return images

    def _arm_action(self, traj, arm, frame, steps):
        rotm = self._frame(traj, f"proprios.{arm}_ee_rotm", frame).reshape(3, 3)
        pos = self._frame(traj, f"proprios.{arm}_ee_pos", frame)
        target_pos = self._future(traj, f"actions.{arm}_ee_pos", frame, steps)
        target_rotm = self._future(
            traj, f"actions.{arm}_ee_rotm", frame, steps
        ).reshape(-1, 3, 3)
        return (
            self._pad((rotm.T @ (target_pos - pos).T).T, steps),
            self._pad(rotm2aa_batch(rotm.T @ target_rotm), steps),
            self._pad(
                self._future(traj, f"actions.{arm}_gripper_pos", frame, steps)
                - self._frame(traj, f"proprios.{arm}_gripper_pos", frame), steps
            ),
        )

    @staticmethod
    def _state(traj, frame):
        fields = {
            f"{side}_{kind}": ObservationTools._frame(
                traj, f"proprios.{side}_{source}", frame
            )
            for side in ("left", "right")
            for kind, source in (("joint", "arm_joint"), ("gripper", "gripper_pos"))
        }
        return compose_state(**fields)

    @staticmethod
    def _crop_bbox(frames, bbox):
        if bbox is None:
            return frames
        top, left, height, width = map(int, bbox)
        cropped = np.zeros(
            (len(frames), height, width, frames.shape[-1]), dtype=frames.dtype
        )
        row_start, row_stop = max(top, 0), min(top + height, frames.shape[1])
        col_start, col_stop = max(left, 0), min(left + width, frames.shape[2])
        if row_stop <= row_start or col_stop <= col_start:
            return cropped
        destination = (
            slice(None),
            slice(row_start - top, row_stop - top),
            slice(col_start - left, col_stop - left),
        )
        source = (slice(None), slice(row_start, row_stop), slice(col_start, col_stop))
        cropped[destination] = frames[source]
        return cropped

    def _augment(self, images):
        from torchvision.transforms.functional import (
            adjust_brightness,
            adjust_contrast,
            adjust_hue,
            adjust_saturation,
        )

        ops = (
            (adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (adjust_contrast, random.uniform(0.5, 1.5)),
            (adjust_saturation, random.uniform(0.5, 1.5)),
            (adjust_hue, random.uniform(0.0, 0.0)),
        )
        flags = [random.randint(0, 1) == 0 for _ in ops]

        out = []
        for image in images:
            image = resize_image(image, factor=32, max_pixels=160000)
            for use, (op, value) in zip(flags, ops):
                if use:
                    image = op(image, value)
            out.append(image)
        return out

    @staticmethod
    def _messages(conversations, images):
        remaining = iter(images)
        assembled = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            fragments = (
                PROMPT_RE.split(turn["value"]) if role == "user" else [turn["value"]]
            )
            content = []
            for fragment in fragments:
                if role == "user" and fragment == "<video>":
                    raise ValueError(
                        "video placeholders are not supported in the JSON dataset"
                    )
                if role == "user" and fragment == "<image>":
                    try:
                        image = next(remaining)
                    except StopIteration:
                        raise ValueError(
                            "number of <image> placeholders exceeds the number of provided images"
                        ) from None
                    content.append({"type": "image", "image": image})
                elif fragment or role == "assistant":
                    content.append({"type": "text", "text": fragment})
            assembled.append({"role": role, "content": content})
        unused = sum(1 for _ in remaining)
        if unused:
            raise ValueError(f"{unused} image(s) remain unused")
        return assembled

    def _delta(self, traj, current_key, target_key, frame, steps):
        return self._pad(
            self._future(traj, target_key, frame, steps)
            - self._frame(traj, current_key, frame),
            steps,
        )

    @staticmethod
    def _frame(traj, key, frame):
        return np.asarray(get_value(traj, key)[frame], dtype=np.float32)

    @staticmethod
    def _future(traj, key, frame, steps):
        return np.asarray(get_value(traj, key)[frame : frame + steps], dtype=np.float32)

    def _pad(self, value, steps):
        window = np.asarray(value)
        missing = self.action_length - steps
        if missing == 0:
            return window
        tail = np.repeat(window[-1:], missing, axis=0)
        return np.concatenate((window, tail), axis=0)
