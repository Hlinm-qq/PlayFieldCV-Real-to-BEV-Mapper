import time

from utils.utils import *
from engine.detection.yolo_engine import *
from engine.tracker.deepsort_engine import *
from engine.detection.preprocessor import *
from engine.detection.postprocessor import *
from engine.detection.hough_detector import *
from engine.tracker.lk_tracker import *
from data_manager.input_manager import *
from data_manager.output_manager import *
from canvas.visualizer import *
from canvas.canvas_utils import *

def tracker_pipeline_v0(config: PipelineConfig, device):
    # Initialize unified stream reader
    stream = UnifiedSource(config.source_path)
    stream.set_max_frame(config.frame_count)
    is_live_stream = isinstance(config.source_path, int) and config.source_path == 0
    is_images = True if stream.get_type() != "video" else False

    # Handler
    tracker = SORTTracker()
    output_manager = OutputManager()
    engine_manager = EngineManager(None, config.model_path)

    # Determine what FPS we are trying to follow
    target_fps = config.target_fps # default not limited
    if not is_images:
        # Acquire the stream FPS
        target_fps = stream.get_fps()

    # Calculate the sleep target
    target_frame_time = (1.0 / target_fps) if target_fps > 0 else 0

    if config.use_wrapper:
        model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO)
    else:
        model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO, is_raw=True)
    model.to(device)
    # print(model.device)

    try:
        # --- PIPELINE LOOP ---
        for frame in stream.get_frame():
            loop_start_time = time.time()
        
            # Inference (Preprocess -> Engine -> Postprocess)
            if not config.enable_tracking:
                if not is_images and not is_live_stream:
                    results = model(frame, stream=True)
                else:
                    results = model(frame)
            else:
                results = model.track(frame, persist=True, conf=0.1)
            print(f"Result size: {len(results)}")

            sort_detections = []
            track_ids = [] 
            for result in results:
                # result extraction
                boxes = result.boxes
        
                if boxes is not None and len(boxes) > 0:
                    xyxy_array = boxes.xyxy.cpu().numpy()
                    xywh_array = boxes.xywh.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy() # confidence score for each box
                    names = [result.names[id.item()] for id in boxes.cls.int().cpu().numpy()] # class name of each box
                    
                    if config.enable_tracking and boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = [0] * confidences.shape[0]

                    # print(f"Boxes shape: {getattr(xywh_array, 'shape', len(xywh_array))}")
                    # print(f"Conf: {len(confidences)}, Names: {len(names)}, Tracks: {len(track_ids)}")
                    for (cx, cy, w, h), conf, name, track_id in zip(xywh_array, confidences, names, track_ids):
                        print(f"Frame object -> Class: {name} | Conf: {conf:.2f} | X:{cx:.1f} Y:{cy:.1f} W:{w:.1f} H:{h:.1f}")
        
                        x_top_left = int(cx - (w / 2))
                        y_top_left = int(cy - (h / 2))
                        text_y = max(15, y_top_left - 10)
        
                        cv2.rectangle(frame, (x_top_left, y_top_left), (x_top_left + int(w), y_top_left + int(h)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name}: {conf:.2f}", (x_top_left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        if config.enable_tracking:
                            cv2.putText(frame, f"ID: {track_id}", (x_top_left, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)           

            # Measure pipeline speed
            process_time = time.time() - loop_start_time
        
            # Throttle, short delay to match target FPS, if applicable
            if target_frame_time > 0 and process_time < target_frame_time:
                sleep_amount = target_frame_time - process_time
                time.sleep(sleep_amount)
        
            # Calculate real final FPS
            total_loop_time = time.time() - loop_start_time
            actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0
        
            # Draw telemetry on screen
            cv2.putText(frame, f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}")
            print(f"Actual FPS: {int(actual_fps)}")

            # Display result
            key_pressed = output_manager.display_frame(frame)
            if not output_manager.get_is_notebook():
                cv2.waitKey(1000 * config.sleep_second)
            else:
                time.sleep(config.sleep_second)
            # cv2.imshow("System Output", frame)
            if key_pressed == ord('q'): break

    except KeyboardInterrupt:
        # This catches the Ctrl+C or force-stop so it doesn't print a massive error
        print_boxed_message("Force stopped. Cleaning up...")
        return False

    finally:
        print_boxed_message("Closing streams and windows...")
        stream.close()
        cv2.destroyAllWindows()
        
    return True

def tracker_pipeline_v1(config: PipelineConfig, device):
    # Initialize unified stream reader
    stream = UnifiedSource(config.source_path)
    stream.set_max_frame(config.frame_count)
    is_live_stream = isinstance(config.source_path, int) and config.source_path == 0
    is_images = True if stream.get_type() != "video" else False

    # Handler
    preprocessor = Preprocessor()
    postprocessor = YOLOPostprocessor()
    tracker = SORTTracker()
    output_manager = OutputManager()
    engine_manager = EngineManager(None, config.model_path)

    # Determine what FPS we are trying to follow
    target_fps = config.target_fps # default not limited
    if not is_images:
        # Acquire the stream FPS
        target_fps = stream.get_fps()

    # Calculate the sleep target
    target_frame_time = (1.0 / target_fps) if target_fps > 0 else 0

    model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO, is_raw=True)
    model.to(device)

    try:
        # --- PIPELINE LOOP ---
        for frame in stream.get_frame():
            loop_start_time = time.time()
        
            # Preprocess
            input_tensor, ratio, dw, dh = preprocessor.process(frame)
            input_tensor = input_tensor.to(device)

            # Forward Pass
            with torch.no_grad():
                raw_output = model(input_tensor)[0]
            print(raw_output.shape)

            # Postprocess 
            Detections = postprocessor.process2(raw_output, ratio, dw, dh, config.confidence_threshold, config.nms_threshold)
            print(f"Detection size: {len(Detections)}")

            active_tracks = tracker.update(Detections)
            
            for track in active_tracks:
                track_id = track['track_id']

                box = track['box']
                print(box)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
            # Measure pipeline speed
            process_time = time.time() - loop_start_time
        
            # Throttle, short delay to match target FPS, if applicable
            if target_frame_time > 0 and process_time < target_frame_time:
                sleep_amount = target_frame_time - process_time
                time.sleep(sleep_amount)
        
            # Calculate real final FPS
            total_loop_time = time.time() - loop_start_time
            actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0
        
            # Draw telemetry on screen
            cv2.putText(frame, f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}")
            print(f"Actual FPS: {int(actual_fps)}")

            # Display result
            key_pressed = output_manager.display_frame(frame)
            if not output_manager.get_is_notebook():
                cv2.waitKey(1000 * config.sleep_second)
            else:
                time.sleep(config.sleep_second)
            # cv2.imshow("System Output", frame)
            if key_pressed == ord('q'): break

    except KeyboardInterrupt:
        # This catches the Ctrl+C or force-stop so it doesn't print a massive error
        print_boxed_message("Force stopped. Cleaning up...")
        return False

    finally:
        print_boxed_message("Closing streams and windows...")
        stream.close()
        cv2.destroyAllWindows()
        
    # Return "finish" flag
    return True

# Bird-eye view MVP
def tracker_pipeline_v2(config: PipelineConfig, device):
    # Standard COCO 17-Keypoint Skeleton Pairs
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Head/Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso and Arms
        (5, 11), (6, 12), (11, 12),               # Hips
        (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
    ]

    COURT_ZONES = {
            "1": {
                "name": "Full Court (Outer Corners)",
                "dst_pts": np.array([[0,0], [1800,0], [0,900], [1800,900]], dtype=np.float32)
            },
            "2": {
                "name": "Left Half (Baseline to Center Line)",
                "dst_pts": np.array([[0,0], [900,0], [0,900], [900,900]], dtype=np.float32)
            },
            "3": {
                "name": "Left Attack Zone (Attack Line to Center Line)",
                "dst_pts": np.array([[600,0], [900,0], [600,900], [900,900]], dtype=np.float32)
            },
            "4": {
                "name": "Right Attack Zone (Center Line to Right Attack Line)",
                "dst_pts": np.array([[900,0], [1200,0], [900,900], [1200,900]], dtype=np.float32)
            },
            "5": {
                "name": "Left Back Zone (Left baseline to left attack line)",
                "dst_pts": np.array([[0, 0], [600, 0], [600, 900], [0, 900]], dtype=np.float32)
            }
        }

    print_config(config)

    # Initialize unified stream reader
    stream = UnifiedSource(config.source_path)
    stream.set_max_frame(config.frame_count)
    is_live_stream = isinstance(config.source_path, int) and config.source_path == 0
    is_images = True if stream.get_type() != "video" else False

    # Determine canvas size
    canvas_w, canvas_h = 400, 800

    # print(stream.get_first_frame())
    # print(get_frame_stats(stream.get_first_frame()))
    # return False

    first_frame = stream.get_first_frame()
    if config.is_manual_picker:
        # Set the source reference points
        point_picker = PointPicker()
        src_points = point_picker.get_points(first_frame)
    else:
        # src_points = np.float32([[305, 373], [995, 376], [1191, 705], [67, 698]])
        src_points = np.float32([[147, 329], [291, 333], [269, 483], [81, 474]])

    frame_stats = get_frame_stats(first_frame)
    dst_points = COURT_ZONES["5"].get("dst_pts")

    if len(src_points) != 4:
        raise ValueError(f"Expected 4 points for 'src_points', but got {len(src_points)}")
        return False
    else:
        print(f"'src_points': {src_points}")
        # return True
    
    # Handler
    tracker = SORTTracker()
    output_manager = OutputManager()
    engine_manager = EngineManager(EngineType.ULTRALYTICSYOLO, config.model_path)
    visualizer = Visualizer(canvas_w, canvas_h, config.map_padding)
    projector = FieldProjector(src_points, canvas_w, canvas_h, dst_points, config.map_padding)
    
    # Determine what FPS we are trying to follow
    target_fps = config.target_fps # default not limited
    if not is_images:
        # Acquire the stream FPS
        target_fps = stream.get_fps()

    # Calculate the sleep target
    target_frame_time = (1.0 / target_fps) if target_fps > 0 else 0

    if config.use_wrapper:
        model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO)
    else:
        model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO, is_raw=True)
    model.to(device)

    if config.save_video and config.frame_count > 5:
        if config.is_combine_result:
            video_width = frame_stats['Dimensions'][0] + canvas_w
            video_height = max(frame_stats['Dimensions'][1], canvas_h)
        
            output_video_fps = target_fps
            if target_fps <= 0:
                output_video_fps = config.sleep_second
            # Initialize video writer
            output_manager.init_video_writer(output_video_fps, video_width, video_height)

    frame_counter = 0
    player_history = {}
    
    try:
        # --- PIPELINE LOOP ---
        for frame in stream.get_frame():
            frame_counter += 1

            loop_start_time = time.time()

            # Get fresh canvas for this frame
            map_canvas = visualizer.draw_empty_court()

            # Inference (Preprocess -> Engine -> Postprocess)
            if not config.enable_tracking:
                if not is_images and not is_live_stream:
                    outputs = model(frame, stream=True, classes=config.class_id_filter)
                else:
                    outputs = model(frame, classes=config.class_id_filter)
            else:
                outputs = model.track(frame, persist=True, conf=0.45, iou=0.5, tracker="botsort.yaml", classes=config.class_id_filter)
            print(f"Result size: {len(outputs)}")

            results = dict_to_list(engine_manager.get_output(outputs, BoxFormat.XYXY | BoxFormat.ID | BoxFormat.KEYPOINT))
            players = []
            
            # print("results")
            # print(results)
            
            # print("outputs")
            # print(outputs)
            # return True

            # Threshold to trust the ankle
            CONF_THRESH = 0.5 

            for result in results:
                track_id = result.get("id", -1) 
                x1, y1, x2, y2 = result.get("xyxy", [0,0,0,0])
                # print('result["xyxy"]')
                # print(result["xyxy"])

                # Calculate foot coordinate (bottom center of bounding box)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                # Pose Estimation
                kpts_xy = result.get("keypoints_xy")
                print(kpts_xy)
                kpts_conf = result.get("keypoints_conf")
                is_occluded = True

                if kpts_xy is not None and kpts_conf is not None:
                    # left (15) and right (16) ankles
                    l_ankle_xy = kpts_xy[15]
                    r_ankle_xy = kpts_xy[16]
                    l_ankle_conf = kpts_conf[15]
                    r_ankle_conf = kpts_conf[16]

                    valid_ankles_x = []
                    valid_ankles_y = []

                    if l_ankle_conf > CONF_THRESH:
                        valid_ankles_x.append(l_ankle_xy[0])
                        valid_ankles_y.append(l_ankle_xy[1])
                    
                    if r_ankle_conf > CONF_THRESH:
                        valid_ankles_x.append(r_ankle_xy[0])
                        valid_ankles_y.append(r_ankle_xy[1])

                    # At least one ankle clearly
                    if len(valid_ankles_x) > 0:
                        foot_x = int(sum(valid_ankles_x) / len(valid_ankles_x))
                        foot_y = int(sum(valid_ankles_y) / len(valid_ankles_y))
                        is_occluded = False

                # Occlusion handling, by history position
                if track_id != -1:
                    if is_occluded and track_id in player_history:
                        foot_x, foot_y = player_history[track_id]
                    else:
                        player_history[track_id] = (foot_x, foot_y)
                
                players.append({
                    "id": track_id,
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (foot_x, foot_y), 5, (255, 0, 0), -1)
                if track_id != -1:
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                # draw pose estimation
                if kpts_xy is not None and kpts_conf is not None:
                    # keypoints
                    for i, (x, y) in enumerate(kpts_xy):
                        conf = kpts_conf[i]
                        if conf > CONF_THRESH: 
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                    # bones (edges)
                    for pt1, pt2 in SKELETON_CONNECTIONS:
                        conf1 = kpts_conf[pt1]
                        conf2 = kpts_conf[pt2]
                        
                        # Only draw the line if BOTH connected points are visible
                        if conf1 > CONF_THRESH and conf2 > CONF_THRESH:
                            x_a, y_a = kpts_xy[pt1]
                            x_b, y_b = kpts_xy[pt2]
                            cv2.line(frame, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (255, 0, 255), 2)

                # Math: Convert real foot coordinate to map coordinate
                map_x, map_y = projector.get_map_coordinate(foot_x, foot_y)
                
                # Draw: Plot it on the 2D canvas
                map_canvas = visualizer.draw_player_on_map(map_canvas, map_x, map_y, track_id)
            
            # Measure pipeline speed
            process_time = time.time() - loop_start_time
        
            # Throttle, short delay to match target FPS, if applicable
            if target_frame_time > 0 and process_time < target_frame_time:
                sleep_amount = target_frame_time - process_time
                time.sleep(sleep_amount)
        
            # Calculate real final FPS
            total_loop_time = time.time() - loop_start_time
            actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0
        
            # Draw telemetry on screen
            cv2.putText(frame, f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}")
            print(f"Actual FPS: {int(actual_fps)}")

            if config.is_combine_result:
                # Glue the two images together horizontally
                combined_view = stitch_two_frames(frame, map_canvas)
                print(get_frame_stats(combined_view))
                # combined_view = cv2.resize(combined_view, (1500, 700)) 

            if frame_counter < 11:
                if not config.is_combine_result:
                    output_manager.save_image(frame, 'output', f"ori_frame_{frame_counter}.jpg")
                    output_manager.save_image(map_canvas, 'output', f"map_frame_{frame_counter}.jpg")
                else:
                    output_manager.save_image(combined_view, 'output', f"combined_frame_{frame_counter}.jpg")

            if config.save_video:
                if config.is_combine_result:
                    output_manager.video_out(combined_view)
            
            # Display result
            if not config.is_combine_result:
                key_pressed = output_manager.display_frame(frame, "Original Detection")
                key_pressed2 = output_manager.display_frame(map_canvas, "2D Radar Map")
                if key_pressed == ord('q'): break
                if key_pressed2 == ord('q'): break
            else:
                key_pressed = output_manager.display_frame(combined_view, "Combined Output")
                if key_pressed == ord('q'): break

            if not output_manager.get_is_notebook():
                cv2.waitKey(1000 * config.sleep_second)
            else:
                time.sleep(config.sleep_second)
    
    except KeyboardInterrupt:
        # Catches the Ctrl+C or force-stop
        print_boxed_message("Force stopped. Cleaning up...")
        return False

    finally:
        print_boxed_message("Closing streams and windows...")
        stream.close()
        output_manager.close()
        cv2.destroyAllWindows()

    return True

def tracker_pipeline_v2_nonpose(config: PipelineConfig, device):
    # Initialize unified stream reader
    stream = UnifiedSource(config.source_path)
    stream.set_max_frame(config.frame_count)
    is_live_stream = isinstance(config.source_path, int) and config.source_path == 0
    is_images = True if stream.get_type() != "video" else False

    # Determine canvas size
    canvas_w, canvas_h = 400, 800

    # print(stream.get_first_frame())
    # print(get_frame_stats(stream.get_first_frame()))
    # return False

    first_frame = stream.get_first_frame()
    if config.is_manual_picker:
        # Set the source reference points
        point_picker = PointPicker()
        src_points = point_picker.get_points(first_frame)
    else:
        src_points = np.float32([[305, 373], [995, 376], [1191, 705], [67, 698]])

    frame_stats = get_frame_stats(first_frame)

    if len(src_points) != 4:
        raise ValueError(f"Expected 4 points for 'src_points', but got {len(src_points)}")
        return False
    else:
        print(f"'src_points': {src_points}")
        # return True
    
    # Handler
    tracker = SORTTracker()
    output_manager = OutputManager()
    engine_manager = EngineManager(EngineType.ULTRALYTICSYOLO, config.model_path)
    visualizer = Visualizer(canvas_w, canvas_h, config.map_padding)
    content_size = visualizer.get_content_size()
    projector = FieldProjector(src_points, canvas_w, canvas_h, config.map_padding)
    
    # Determine what FPS we are trying to follow
    target_fps = config.target_fps
    if not is_images:
        # Acquire the stream FPS
        target_fps = stream.get_fps()

    # Calculate sleep target
    target_frame_time = (1.0 / target_fps) if target_fps > 0 else 0

    if config.use_wrapper:
        model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO)
    else:
        model = engine_manager.load_model(EngineType.ULTRALYTICSYOLO, is_raw=True)
    model.to(device)

    if config.save_video and config.frame_count > 5:
        if config.is_combine_result:
            video_width = frame_stats['Dimensions'][0] + canvas_w
            video_height = max(frame_stats['Dimensions'][1], canvas_h)
        
            output_video_fps = target_fps
            if target_fps <= 0:
                output_video_fps = config.sleep_second
            # Initialize video writer
            output_manager.init_video_writer(output_video_fps, video_width, video_height)

    frame_counter = 0
    
    try:
        # --- PIPELINE LOOP ---
        for frame in stream.get_frame():
            frame_counter += 1

            loop_start_time = time.time()

            # Get fresh canvas for this frame
            map_canvas = visualizer.draw_empty_court()

            # Inference (Preprocess -> Engine -> Postprocess)
            if not config.enable_tracking:
                if not is_images and not is_live_stream:
                    outputs = model(frame, stream=True, classes=config.class_id_filter)
                else:
                    outputs = model(frame, classes=config.class_id_filter)
            else:
                outputs = model.track(frame, persist=True, conf=0.1, classes=config.class_id_filter)
            print(f"Result size: {len(outputs)}")

            results = dict_to_list(engine_manager.get_output(outputs, BoxFormat.XYXY))
            players = []
            
            # print("results")
            # print(results)
            
            # print("outputs")
            # print(outputs)
            # return True

            for result in results:
                x1, y1, x2, y2 = result["xyxy"]
                # print('result["xyxy"]')
                # print(result["xyxy"])
                # Calculate foot coordinate (bottom center of bounding box)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                players.append({
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (foot_x, foot_y), 5, (255, 0, 0), -1)

                # Math: Convert real foot coordinate to map coordinate
                map_x, map_y = projector.get_map_coordinate(foot_x, foot_y)
                
                # Draw: Plot it on the 2D canvas
                map_canvas = visualizer.draw_player_on_map(map_canvas, map_x, map_y, -1)
            
            # Measure pipeline speed
            process_time = time.time() - loop_start_time
        
            # Throttle, short delay to match target FPS, if applicable
            if target_frame_time > 0 and process_time < target_frame_time:
                sleep_amount = target_frame_time - process_time
                time.sleep(sleep_amount)
        
            # Calculate real final FPS
            total_loop_time = time.time() - loop_start_time
            actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0
        
            # Draw telemetry on screen
            cv2.putText(frame, f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}")
            print(f"Actual FPS: {int(actual_fps)}")

            if config.is_combine_result:
                # Glue the two images together horizontally
                combined_view = stitch_two_frames(frame, map_canvas)
                print(get_frame_stats(combined_view))
                # combined_view = cv2.resize(combined_view, (1500, 700)) 

            if frame_counter < 11:
                if not config.is_combine_result:
                    output_manager.save_image(frame, 'output', f"ori_frame_{frame_counter}.jpg")
                    output_manager.save_image(map_canvas, 'output', f"map_frame_{frame_counter}.jpg")
                else:
                    output_manager.save_image(combined_view, 'output', f"combines_frame_{frame_counter}.jpg")

            if config.save_video:
                if config.is_combine_result:
                    output_manager.video_out(combined_view)
            
            # Display result
            if not config.is_combine_result:
                key_pressed = output_manager.display_frame(frame, "Original Detection")
                key_pressed2 = output_manager.display_frame(map_canvas, "2D Radar Map")
                if key_pressed == ord('q'): break
                if key_pressed2 == ord('q'): break
            else:
                key_pressed = output_manager.display_frame(combined_view, "Combined Output")
                if key_pressed == ord('q'): break

            if not output_manager.get_is_notebook():
                cv2.waitKey(1000 * config.sleep_second)
            else:
                time.sleep(config.sleep_second)
    
    except KeyboardInterrupt:
        # Catches the Ctrl+C or force-stop
        print_boxed_message("Force stopped. Cleaning up...")
        return False

    finally:
        print_boxed_message("Closing streams and windows...")
        stream.close()
        output_manager.close()
        cv2.destroyAllWindows()

        if output_manager.get_is_notebook():
            cv2.waitKey(1)

    return True

# Math based Edge detector Method
def tracker_pipeline_v3(config: PipelineConfig, device):
    print_config(config)

    # Scale: 100 pixels = 1 meter
    # Canvas Size: 1800w x 900h
    # Center Line X = 900
    # Left Attack Line X = 600 (900 - 300)
    # Right Attack Line X = 1200 (900 + 300)

    COURT_ZONES = {
        "1": {
            "name": "Full Court (Outer Corners)",
            "dst_pts": np.array([[0,0], [1800,0], [0,900], [1800,900]], dtype=np.float32)
        },
        "2": {
            "name": "Left Half (Baseline to Center Line)",
            "dst_pts": np.array([[0,0], [900,0], [0,900], [900,900]], dtype=np.float32)
        },
        "3": {
            "name": "Left Attack Zone (Attack Line to Center Line)",
            "dst_pts": np.array([[600,0], [900,0], [600,900], [900,900]], dtype=np.float32)
        },
        "4": {
            "name": "Right Attack Zone (Center Line to Right Attack Line)",
            "dst_pts": np.array([[900,0], [1200,0], [900,900], [1200,900]], dtype=np.float32)
        },
        "5": {
            "name": "Left Back Zone (Left baseline to left attack line)",
            "dst_pts": np.array([[0, 0], [600, 0], [600, 900], [0, 900]], dtype=np.float32)
        }
    }

    # Initialize unified stream reader
    stream = UnifiedSource(config.source_path)
    stream.set_max_frame(config.frame_count)
    is_live_stream = isinstance(config.source_path, int) and config.source_path == 0
    is_images = True if stream.get_type() != "video" else False

    canvas_w, canvas_h = 1800, 900
    dst_points = COURT_ZONES["5"].get("dst_pts")

    first_frame = stream.get_first_frame()
    if config.is_manual_picker:
        # Set the source reference points
        point_picker = PointPicker()
        src_points = point_picker.get_points(first_frame)
    else:
        # src_points = np.float32([[305, 373], [995, 376], [1191, 705], [67, 698]])
        src_points = np.float32([[147, 329], [291, 333], [269, 483], [81, 474]])

    if len(src_points) != 4:
        raise ValueError(f"Expected 4 points for 'src_points', but got {len(src_points)}")
        return False
    else:
        print(f"'src_points': {src_points}")

    # --- Handlers initialization ---
    output_manager = OutputManager()
    court_detector = HoughDetector()
    engine_manager = EngineManager(EngineType.ULTRALYTICSYOLO, config.model_path)
    projector = FieldProjector(src_points, canvas_w, canvas_h, dst_points, config.map_padding)
    visualizer = Visualizer(canvas_w, canvas_h, config.map_padding, config.map_orientation)
    tracker = LKTracker()
    tracker.initialize(first_frame, src_points, dst_points)

    # Determine what FPS we are trying to follow
    default_fps = config.default_fps if config.default_fps > 0 else 30
    target_fps = config.target_fps
    if not is_images:
        # Acquire the stream FPS
        target_fps = stream.get_fps()
    target_fps = target_fps if target_fps > 0 else default_fps

    # Calculate sleep target
    target_frame_time = (1.0 / target_fps)

    if config.use_wrapper:
        model = engine_manager.load_model(None)
    else:
        model = engine_manager.load_model(None, is_raw=True)
    model.to(device)

    frame_stats = get_frame_stats(first_frame)
    if config.save_video and config.frame_count > 5:
        if config.is_combine_result:
            video_width = frame_stats['Dimensions'][0] + canvas_w
            video_height = max(frame_stats['Dimensions'][1], canvas_h)
        
            output_video_fps = target_fps
            if target_fps <= 0:
                output_video_fps = config.sleep_second
            # Initialize video writer
            output_manager.init_video_writer(output_video_fps, video_width, video_height)


    # "dst_pts": np.array([[0, 0], [600, 0], [0, 900], [600, 900]], dtype=np.float32)
    current_H, _ = cv2.findHomography(src_points, dst_points)

    frame_counter = 0
    try:
        # --- PIPELINE LOOP ---
        for frame in stream.get_frame():
            frame_counter += 1
            # print(frame_counter)
            loop_start_time = time.time()

            # Optical Flow (LK) 
            lk_matrix, lk_ok = tracker.update_matrix(frame)
            tracker.debug_draw_lk_tracker(frame, tracker.old_points)

            if lk_ok:
                current_H = lk_matrix # Continuous tracking

            # Get fresh canvas for this frame
            map_canvas = visualizer.draw_empty_court()

            # Inference (Preprocess -> Engine -> Postprocess)
            if not config.enable_tracking:
                if not is_images and not is_live_stream:
                    outputs = model(frame, stream=True, classes=config.class_id_filter)
                else:
                    outputs = model(frame, classes=config.class_id_filter)
            else:
                outputs = model.track(frame, persist=True, conf=0.1, classes=config.class_id_filter)
            print(f"Detection Output size: {len(outputs)}")
            # print(f"outputs")
            # print(outputs)

            results = dict_to_list(engine_manager.get_output(outputs, BoxFormat.XYXY))
            players = []

            for result in results:
                x1, y1, x2, y2 = result["xyxy"]
                print('result["xyxy"]')
                print(result["xyxy"])
                # Calculate foot coordinate (bottom center of bounding box)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                players.append({
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (foot_x, foot_y), 5, (255, 0, 0), -1)

                # Math: Convert real foot coordinate to map coordinate
                map_x, map_y = projector.get_map_coordinate(foot_x, foot_y, current_H)
                
                if map_x is not None:
                    # Draw: Plot it on the 2D canvas
                    # map_canvas = visualizer.draw_player_on_map(map_canvas, map_x, map_y, -1)
                    visualizer.draw_player_on_map(map_canvas, map_x, map_y, -1)
                
            if frame_counter % config.recalibration_rate == 0 or not lk_ok:
                if frame_counter >= (config.recalibration_rate*2):
                    frame_counter = 0

                raw_lines = court_detector.supercharged_line_detection(frame)

                if raw_lines is not None:
                    # Ghost Matcher sorts them into buckets
                    categorized_buckets = court_detector.match_lines_to_ghost_court(raw_lines, current_H)
                    
                    # Fit the grouped detected lines 
                    final_lines = court_detector.fit_perfect_lines(categorized_buckets)
                    court_detector.debug_draw_fit_lines(frame, final_lines)
                    
                    # Extract new detected corners
                    detected_corner, dst_pts = court_detector.extract_dynamic_corners(final_lines)
                    court_detector.debug_draw_corners(frame, detected_corner)
                    
                    # Feeding to the next cycle
                    if detected_corner is not None:
                        print("Recalibration Successful! Snapping matrix and tracker...")
                        # print(detected_corner)
                        
                        try:
                            # Update the Homography Matrix back to detected edges
                            # current_H, _ = cv2.findHomography(detected_corner, dst_pts)
                            current_H = cv2.getPerspectiveTransform(detected_corner, dst_pts)

                            # Overwrite LK's drifting
                            tracker.initialize(frame, detected_corner, dst_pts)
                        except cv2.error as e:
                            print(f"OpenCV Error caught: {e}")

            # Measure pipeline speed
            process_time = time.time() - loop_start_time
        
            # Throttle, short delay to match target FPS, if applicable
            if target_frame_time > 0 and process_time < target_frame_time:
                sleep_amount = target_frame_time - process_time
                time.sleep(sleep_amount)

            # Calculate real final FPS
            total_loop_time = time.time() - loop_start_time
            actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0
                    
            # Draw telemetry on screen
            cv2.putText(frame, f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}")
            print(f"Actual FPS: {int(actual_fps)}")

            if config.is_combine_result:
                # Glue the two images together horizontally
                combined_view = stitch_two_frames(frame, map_canvas)
                print(get_frame_stats(combined_view))
                # combined_view = cv2.resize(combined_view, (1500, 700)) 
            
            # Display result
            if not config.is_combine_result:
                key_pressed = output_manager.display_frame(frame, "Original Detection")
                key_pressed2 = output_manager.display_frame(map_canvas, "2D Map")
                if key_pressed == ord('q'): break
                if key_pressed2 == ord('q'): break
            else:
                key_pressed = output_manager.display_frame(combined_view, "Combined Output")
                if key_pressed == ord('q'): break

            if not output_manager.get_is_notebook():
                cv2.waitKey(1000 * config.sleep_second)
            else:
                time.sleep(config.sleep_second)

    except KeyboardInterrupt:
        # Catches the Ctrl+C or force-stop
        print_boxed_message("Force stopped. Cleaning up...")
        return False

    finally:
        print_boxed_message("Closing streams and windows...")
        stream.close()
        output_manager.close()
        cv2.destroyAllWindows()
   
    return True

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())

    print(torch.cuda.is_available())  # True if GPU detected

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    archive_dir = get_path("../dataset")
    pascal_2012_dir = archive_dir / get_path("VOC2012/JPEGImages")
    video_frame_dir = archive_dir / get_path("video_frame/people/images")
    volley_dir = archive_dir / get_path("volleyball_3/train/images")

    # ==========================================================
    # target_source1 = volley_dir
    # check_path(target_source1)
    # config_v0 = PipelineConfig(source_path=str(target_source1),
    #                         use_wrapper=True,
    #                         enable_tracking=False)
    # is_finished = tracker_pipeline_v0(config_v0, device)

    # ==========================================================
    # target_source2 = video_frame_dir
    # check_path(target_source2)
    # config_v1 = PipelineConfig(source_path=str(target_source2))
    # is_finished = tracker_pipeline_v1(config_v1, device)

    # ==========================================================
    # target_source2 = volley_dir
    # # target_classes = None
    # target_classes = [0, 32]
    # check_path(target_source2)
    # config = PipelineConfig(source_path=str(target_source2),
    #                         use_wrapper=True,
    #                         enable_tracking=False,
    #                         is_combine_result=True, 
    #                         save_video=False,
    #                         map_padding=5, 
    #                         class_id_filter=target_classes,
    #                         is_manual_picker=False,
    #                         frame_count=2, 
    #                         model_path="yolov8s.pt")
    # is_finished = tracker_pipeline_v2(config, device)

    # ==========================================================
    target_source3 = volley_dir
    # target_classes = None
    target_classes = [0, 32]
    check_path(target_source3)
    config = PipelineConfig(source_path=str(target_source3),
                            use_wrapper=True,
                            enable_tracking=False,
                            is_combine_result=True, 
                            save_video=False,
                            map_padding=5, 
                            class_id_filter=target_classes,
                            is_manual_picker=False,
                            frame_count=5, 
                            model_path="yolov8s.pt",
                            sleep_second=5,
                            map_orientation="vertical",
                            recalibration_rate=2)
    is_finished = tracker_pipeline_v3(config, device)

    return is_finished

if __name__ == "__main__":
    result = main()
    if result:
        print(f"Pipeline ran successfully")
    else:
        print(f"Pipeline failed")

    # parent_dir = Path("..") 
    # dataset_dir = parent_dir / "dataset"

    # target_dir = dataset_dir / "volleyball_3" / "train" / "images"
    # rename_media(target_dir, "frame", MediaFilter.IMAGES, overwrite=True, pattern="clip_four*", start_num=427)
    None