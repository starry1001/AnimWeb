---
license: mit
library_name: ultralytics
tags:
  - object-detection
  - yolo
  - gui
  - ui-detection
  - omniparser
pipeline_tag: object-detection
---

# GPA-GUI-Detector

A YOLO-based GUI element detection model for detecting interactive UI elements (icons, buttons, etc.) on screen for GUI Process Automation. This model is finetuned from the [OmniParser](https://github.com/microsoft/OmniParser) ecosystem.

## Model

The model weight file is `model.pt`. It is a YOLO model trained with the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.

## Installation

```bash
pip install ultralytics
```

## Usage

### Basic Inference

```python
from ultralytics import YOLO

model = YOLO("model.pt")
results = model("screenshot.png")
```

### Detection with Custom Parameters

```python
from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO("model.pt")

# Run inference with custom confidence and image size
results = model.predict(
    source="screenshot.png",
    conf=0.05,        # confidence threshold
    imgsz=640,        # input image size
    iou=0.7,          # NMS IoU threshold
)

# Parse results
boxes = results[0].boxes.xyxy.cpu().numpy()   # bounding boxes in [x1, y1, x2, y2]
scores = results[0].boxes.conf.cpu().numpy()  # confidence scores

# Draw results on image
img = Image.open("screenshot.png")
for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = box
    print(f"Detected UI element at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] (conf: {score:.2f})")

# Or save the annotated image directly
results[0].save("result.png")
```

### Integration with OmniParser

```python
import sys
sys.path.append("/path/to/OmniParser")

from util.utils import get_yolo_model, predict_yolo
from PIL import Image

model = get_yolo_model("model.pt")
image = Image.open("screenshot.png")

boxes, confidences, phrases = predict_yolo(
    model=model,
    image=image,
    box_threshold=0.05,
    imgsz=640,
    scale_img=False,
    iou_threshold=0.7,
)

for i, (box, conf) in enumerate(zip(boxes, confidences)):
    print(f"Element {i}: box={box.tolist()}, confidence={conf:.2f}")
```

## Example

Detection results on a sample screenshot (1920x1080) from the [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding) benchmark (`conf=0.05`, `iou=0.1`, `imgsz=1280`).

**Input Screenshot**

<p align="center">
  <img src="images/example_input.png" width="80%" alt="Input Screenshot"/>
</p>

<table>
  <tr>
    <th align="center">OmniParser V2</th>
    <th align="center">GPA-GUI-Detector</th>
  </tr>
  <tr>
    <td align="center"><img src="images/example_omniparser.png" width="92%" alt="OmniParser V2"/></td>
    <td align="center"><img src="images/example_gpa.png" width="99%" alt="GPA-GUI-Detector"/></td>
  </tr>
</table>

## License

This model is released under the [MIT License](https://opensource.org/licenses/MIT).
