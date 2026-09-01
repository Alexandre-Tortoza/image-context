"""ROS1 bag adapter that decodes only selected RGB messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from image_context.models import ImageSample
from image_context.sampling import random_sample_indices


class RosbagImageSampler:
    """Randomly sample an image topic without loading all images into memory."""

    def __init__(self, bag_path: Path, image_topic: str) -> None:
        self._bag_path = bag_path
        self._image_topic = image_topic
        self._typestore = get_typestore(Stores.ROS1_NOETIC)

    def sample(self, count: int, seed: int, destination: Path) -> tuple[ImageSample, ...]:
        """Select indices from connection metadata, then decode them in one forward pass."""
        if not self._bag_path.is_file():
            raise FileNotFoundError(f"ROS bag not found: '{self._bag_path}'.")
        with Reader(self._bag_path) as reader:
            connection = next(
                (item for item in reader.connections if item.topic == self._image_topic), None
            )
            if connection is None:
                raise ValueError(
                    f"Image topic '{self._image_topic}' was not found in '{self._bag_path}'."
                )
            selected = random_sample_indices(connection.msgcount, count, seed)
            selected_set = set(selected)
            samples: list[ImageSample] = []
            for index, (_, bag_timestamp, raw_data) in enumerate(
                reader.messages(connections=[connection])
            ):
                if index not in selected_set:
                    continue
                message = self._typestore.deserialize_ros1(raw_data, connection.msgtype)
                pixels = _decode_image(message)
                frame_id = f"frame-{index:06d}"
                frame_directory = destination / frame_id
                frame_directory.mkdir(parents=True, exist_ok=True)
                image_path = frame_directory / "image.png"
                if not cv2.imwrite(str(image_path), pixels):
                    raise OSError(f"Could not write extracted image to '{image_path}'.")
                height, width = pixels.shape[:2]
                samples.append(
                    ImageSample(
                        frame_id=frame_id,
                        source_index=index,
                        timestamp_ns=bag_timestamp,
                        width=width,
                        height=height,
                        image_path=image_path,
                    )
                )
                if len(samples) == count:
                    break
        if len(samples) != count:
            raise RuntimeError(f"Selected {count} images but extracted only {len(samples)}.")
        return tuple(samples)


def _decode_image(message: Any) -> NDArray[np.uint8]:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    raw = np.asarray(message.data, dtype=np.uint8).reshape(height, step)
    if encoding in {"bgr8", "rgb8"}:
        pixels = raw[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            return pixels.copy()
        return cast(NDArray[np.uint8], cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if encoding in {"bgra8", "rgba8"}:
        pixels = raw[:, : width * 4].reshape(height, width, 4)
        conversion = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cast(NDArray[np.uint8], cv2.cvtColor(pixels, conversion))
    if encoding in {"mono8", "8uc1"}:
        mono = raw[:, :width]
        return cast(NDArray[np.uint8], cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR))
    raise ValueError(f"Unsupported ROS image encoding: '{message.encoding}'.")
