import cv2
import numpy as np

class YOLOPostprocessor:
    def process(self, output_tensor, ratio, dw, dh, conf_thresh=0.5, nms_thresh=0.4):
        # YOLO output_tensor: (1, 84, 8400)
        # 8400 bbox
        # 84 columns: (cx, cy, w, h) + 80 class probabilities
        predictions = output_tensor.squeeze().cpu().numpy().T # (8400, 84)

        boxes = []
        bboxes = []
        scores = []
        class_ids = []

        for row in predictions:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]

            if score > conf_thresh:
                # Raw box coordinates
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                # Calculate original scales
                cx = (cx - dw) / ratio
                cy = (cy - dh) / ratio
                w = w / ratio
                h = h / ratio

                # Convert from (center_x, center_y, width, height) 
                # to (top_left_x, top_left_y, width, height) for OpenCV
                x1 = int(cx - (w / 2))
                y1 = int(cy - (h / 2))
                x2 = int(cx + (w / 2))
                y2 = int(cy + (h / 2))
                
                boxes.append([x1, y1, int(w), int(h)])
                bboxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)

        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append({
                    # "box": boxes[i], # [x1, y1, w, h]
                    "box": bboxes[i], # [x1, y1, x2, y2]
                    "score": scores[i], # float
                    "class_id": class_ids[i] # int
                })
                
        return final_detections