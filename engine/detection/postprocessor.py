import cv2
import torch
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

    def process2(self, output_tensor, ratio, dw, dh, conf_thresh=0.5, nms_thresh=0.4, target_class_id=[]):
        # output_tensor: (1, 84, 8400)
        # 8400 bbox
        # 84 columns: (cx, cy, w, h) + 80 class probabilities
        predictions = output_tensor.squeeze().T # (8400, 84)

        # Get boxes and classes scores
        boxes = predictions[:, 0:4]
        class_scores = predictions[:, 4:]

        # Get the highest score and its corresponding class ID for every row
        # max_scores shape: (8400,), class_ids shape: (8400,)
        max_scores, class_ids = torch.max(class_scores, dim=1) 

        # Filter out low confidence
        valid_mask = max_scores > conf_thresh

        # target class mask
        if len(target_class_id) > 0:
            target_tensor = torch.tensor(target_class_id, device=class_ids.device)
            # Check which class_ids match requested targets
            class_mask = torch.isin(class_ids, target_tensor)
            
            # Combine the masks
            valid_mask = valid_mask & class_mask

        # Apply the final mask
        filtered_boxes = boxes[valid_mask]
        filtered_scores = max_scores[valid_mask]
        filtered_class_ids = class_ids[valid_mask]

        cx = filtered_boxes[:, 0]
        cy = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]

        # Calculate original scales
        cx = (cx - dw) / ratio
        cy = (cy - dh) / ratio
        w = w / ratio
        h = h / ratio

        x1 = cx - (w / 2)
        y1 = cy - (h / 2)
        x2 = cx + (w / 2)
        y2 = cy + (h / 2)

        # Move data back to CPU
        x1_np = x1.cpu().numpy().astype(int)
        y1_np = y1.cpu().numpy().astype(int)
        x2_np = x2.cpu().numpy().astype(int)
        y2_np = y2.cpu().numpy().astype(int)
        w_np = w.cpu().numpy().astype(int)
        h_np = h.cpu().numpy().astype(int)

        final_boxes = np.column_stack((x1_np, y1_np, w_np, h_np)).tolist()
        final_bboxes = np.column_stack((x1_np, y1_np, x2_np, y2_np)).tolist()
        final_scores = filtered_scores.cpu().numpy().tolist()
        
        final_classes = filtered_class_ids.cpu().numpy().astype(int).tolist()
        
        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(final_boxes, final_scores, conf_thresh, nms_thresh)

        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append({
                    # "box": boxes[i], # [x1, y1, w, h]
                    "box": final_bboxes[i], # [x1, y1, x2, y2]
                    "score": final_scores[i], # float
                    "class_id": final_classes[i] # int
                })
                
        return final_detections
