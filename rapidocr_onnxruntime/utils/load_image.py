# -*- encoding: utf-8 -*-
# @Author: MTF / rinor4ever
# @Contact: r4ajeti@gmail.com
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class LoadImage:
    def __init__(self):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = self.img_to_ndarray(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = self.img_to_ndarray(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return self.img_to_ndarray(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def img_to_ndarray(self, img: Image.Image) -> np.ndarray:
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)

    def convert_img(self, img: np.ndarray, origin_img_type: Any) -> np.ndarray:
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel BGR

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return np.stack([img[:, :, 0]] * 3, axis=-1)  # Convert grayscale to 3-channel BGR

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return img[:, :, ::-1]  # Convert RGB to BGR
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = np.stack([img_gray] * 3, axis=-1)  # Convert grayscale to 3-channel BGR

        img_alpha = img[..., 1]
        not_a = 255 - img_alpha
        not_a = np.stack([not_a] * 3, axis=-1)  # Convert grayscale to 3-channel BGR

        new_img = np.where(img_alpha[..., np.newaxis] > 0, img_bgr, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        new_img = np.stack([b, g, r], axis=-1)  # Convert RGBA to BGR

        not_a = 255 - a
        not_a = np.stack([not_a] * 3, axis=-1)  # Convert grayscale to 3-channel BGR

        new_img = np.where(a[..., np.newaxis] > 0, new_img, not_a)

        mean_color = np.mean(new_img)
        if mean_color <= 0.0:
            new_img = new_img + not_a
        else:
            new_img = 255 - new_img
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass