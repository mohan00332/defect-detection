import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "Bolt.pt"
CONF = 0.5
MIN_BOX_AREA_RATIO = 0.002
_model = None
_use_filter = None
_target_keyword = "bolt"
_category_keywords = ("nut", "bolt", "gear")
_quality_keywords = ("defect", "non", "non_defect", "good", "ok", "pass", "normal")


def _resolve_model_path():
    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(here, "models")
    candidate = os.path.join(model_dir, MODEL_PATH)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(here, MODEL_PATH)


def get_model():
    global _model
    if _model is None:
        _model = YOLO(_resolve_model_path())
    return _model


def _is_defect(name: str) -> bool:
    name = name.lower()
    if "non_defect" in name or "non-defect" in name:
        return False
    return ("defect" in name) or ("bad" in name) or ("fault" in name)


def _passes_filters(class_name: str, score: float, frame, x1: int, y1: int, x2: int, y2: int, min_conf: float) -> bool:
    name_l = str(class_name).lower()
    if _use_filter and _target_keyword not in name_l:
        if not any(k in name_l for k in _quality_keywords):
            return False
    if any(k in name_l for k in _category_keywords) and _target_keyword not in name_l:
        return False
    if score < min_conf:
        return False
    if MIN_BOX_AREA_RATIO > 0:
        h, w = frame.shape[:2]
        min_area = float(w * h) * MIN_BOX_AREA_RATIO
        box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if box_area < min_area:
            return False
    return True


def detect_frame(frame, conf: float = None, iou: float = None, label_mode: str = "confidence", draw_counts: bool = False):
    global _use_filter
    model = get_model()
    conf = CONF if conf is None else conf
    if _use_filter is None:
        try:
            names = [str(v).lower() for v in model.names.values()]
            _use_filter = any(_target_keyword in name for name in names)
        except Exception:
            _use_filter = False
    if iou is None:
        results = model(frame, conf=conf)
    else:
        results = model(frame, conf=conf, iou=iou)

    detected = 0
    defect = 0

    for box in results[0].boxes:
        detected += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if not _passes_filters(class_name, score, frame, x1, y1, x2, y2, conf):
            continue

        if label_mode == "none":
            label = ""
        elif label_mode == "label":
            label = f"{class_name}"
        else:
            label = f"{class_name} {score:.2f}"

        if _is_defect(class_name):
            defect += 1
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if draw_counts:
        good = max(detected - defect, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        x = 10
        y1 = 30
        y2 = 60
        cv2.putText(frame, f"Defect: {defect}", (x, y1), font, scale, (0, 0, 255), thickness)
        cv2.putText(frame, f"Non-Defect: {good}", (x, y2), font, scale, (0, 255, 0), thickness)

    return frame, detected, defect


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not opening!")
        raise SystemExit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _, _ = detect_frame(frame)
        cv2.imshow("Defect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
