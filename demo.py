import argparse
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn, Tensor
from torchvision.transforms import functional as f
from torchvision import models

logging.basicConfig(
    format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s",
    level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:

    @staticmethod
    def preprocess(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = image.resize((224, 224))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height)

    @staticmethod
    def run(classifier) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        classifier : TorchVisionModel
            Classifier model
        """

        cap = cv2.VideoCapture(0)
        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame, size = Demo.preprocess(frame)
                with torch.no_grad():
                    output = classifier(processed_frame)
                output = list(map(float, nn.functional.softmax(output.view((-1,)))))
                output.append(0.99)
                label = np.array(output).argmax()
                m = max(output)

                cv2.putText(frame,
                            f'{class_emojis[int(label)]}, {m:.3f}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255),
                            thickness=3)
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}",
                            (30, 30), FONT, 1, (255, 0, 255), 2)
                cnt += 1

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo full frame classification...")

    known_args, _ = parser.parse_known_args(params)
    return known_args


class_names = ['like', 'dislike', 'ok']
# class_emojis = ['👍', '👎', '👌']
class_emojis = ['like', 'dislike', 'ok', 'nothing']
classes = list(range(len(class_names)))
class_lookup = {n: i for n, i in zip(class_names, classes)}
num_classes = len(class_names)


def mk_mobilenet(pretrained=True):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    transforms = weights.transforms(antialias=True)
    m = models.mobilenet_v3_large(weights=weights if pretrained else None)
    m.features.requires_grad = False
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m, transforms


if __name__ == "__main__":
    args = parse_arguments()
    w = torch.load('models/mobilenet/examples-all/mobilenetv3-large-final.zip')
    model, transforms = mk_mobilenet(pretrained=False)
    model.load_state_dict(w)
    model.eval()
    if model is not None:
        Demo.run(model)