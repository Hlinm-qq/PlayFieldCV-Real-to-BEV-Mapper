import time

from utils.utils import *
from engine.engine import *
from engine.sink import *
from engine.preprocessor import *
from engine.postprocessor import *
from data_manager.input_manager import *
from data_manager.output_manager import *

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

        if output_manager.get_is_notebook():
            cv2.waitKey(1)
        
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
            Detections = postprocessor.process(raw_output, ratio, dw, dh, config.confidence_threshold, config.nms_threshold)
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

        if output_manager.get_is_notebook():
            cv2.waitKey(1)
        
    # Return "finish" flag
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
    volley_dir = archive_dir / get_path("volley_ball_2/train/images")

    # target_source1 = volley_dir
    # check_path(target_source1)
    # config_v0 = PipelineConfig(source_path=str(target_source1),
    #                         use_wrapper=True,
    #                         enable_tracking=False)
    # is_finished = tracker_pipeline_v0(config_v0, device)

    target_source2 = video_frame_dir
    check_path(target_source2)
    config_v1 = PipelineConfig(source_path=str(target_source2))
    is_finished = tracker_pipeline_v1(config_v1, device)

    return is_finished

if __name__ == "__main__":
    result = main()
    if result:
        print(f"Pipeline ran successfully")
    else:
        print(f"Pipeline failed")

    # parent_dir = Path("..") 
    # dataset_dir = parent_dir / "dataset"

    # target_dir = dataset_dir / "volley_ball_2" / "train" / "images"
    # rename_media(target_dir, "frame", MediaFilter.IMAGES, overwrite=True)
    None