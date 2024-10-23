import cv2
from ultralytics import YOLO

def inference_pipeline(model_path: str, image_path: str, conf_thresh: float = 0.25):
  model = YOLO(model_path)

  results = model.predict(
      source = image_path,
      conf = conf_thresh,
      iou = 0.45,
      show = True,
      save = True
  )
  return results

def process_results(results, image, classes):
  processed_image = image.copy()

  for result in results:
    boxes = result.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      conf = box.conf[0]
      cls_id = int(box.cls[0])
      cls_name = classes[cls_id]


      cv2.rectangle(processed_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0), 2)

      cv2.putText(processed_image,
                  f"{cls_name} {conf:.2f}",
                  (int(x1), int(y1 - 10)),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.9, (0, 255, 0), 2)



    return processed_image
