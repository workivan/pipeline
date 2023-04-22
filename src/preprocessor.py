from typing import Tuple

import cv2
import numpy as np

from src.dataloader_dataset import Batch


class Preprocessor:
    def __init__(self,
                 img_size: Tuple[int, int],
                 padding: int = 0,
                 dynamic_width: bool = False) -> None:
        assert not (padding > 0 and not dynamic_width)

        self.img_size = img_size
        self.padding = padding
        self.dynamic_width = dynamic_width

    def process_img(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float)

        if self.dynamic_width:
            ht = self.img_size[1]
            h, w = img.shape
            f = ht / h
            wt = int(f * w + self.padding)
            wt = wt + (4 - wt) % 4
            tx = (wt - w * f) / 2
            ty = 0
        else:
            wt, ht = self.img_size
            h, w = img.shape
            f = min(wt / w, ht / h)
            tx = (wt - w * f) / 2
            ty = (ht - h * f) / 2

        M = np.float32([[f, 0, tx], [0, f, ty]])
        target = np.ones([ht, wt]) * 255
        img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

        img = cv2.transpose(img)

        img = img / 255 - 0.5
        return img

    def process_batch(self, batch: Batch) -> Batch:
        res_imgs = [self.process_img(img) for img in batch.imgs]
        max_text_len = res_imgs[0].shape[0] // 4
        return Batch(res_imgs, batch.gt_texts, batch.batch_size)
