import time
import random, numpy as np, torch, cv2
import argparse
import random
import sys
from pathlib import Path

from utils.utils import *
from config.volleyball import *
from manager import *

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Player tracking and bird's-eye-view court mapping for volleyball stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("input / output")
    io.add_argument("--source", default="",
                    help="Video file, image folder, RTSP/HTTP URL, or webcam index (e.g. 0)")
    io.add_argument("--frame-count", type=int, default=80, help="Number of frames to process")
    io.add_argument("--save-video", action="store_false", help="Write the output video to disk")

    det = parser.add_argument_group("detection & tracking")
    det.add_argument("--model", default="yolov8s.pt", help="YOLO weights path or name")
    det.add_argument("--classes", type=int, nargs="+", default=[0],
                     help="Class IDs to keep (COCO 0 = person)")
    det.add_argument("--all-classes", action="store_true", help="Keep all classes (overrides --classes)")
    det.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or 'cuda:N'")
    det.add_argument("--detect", action=argparse.BooleanOptionalAction, default=True,
                     help="Run player detection")
    det.add_argument("--tracking", action=argparse.BooleanOptionalAction, default=True,
                     help="Enable ByteTrack ID tracking")
    det.add_argument("--wrapper", action=argparse.BooleanOptionalAction, default=True,
                     help="Use the Ultralytics API wrapper instead of custom post-processing")

    court = parser.add_argument_group("court calibration")
    court.add_argument("--manual-picker", action="store_true",
                       help="Select court corners manually on the first frame")
    court.add_argument("--recalibration-rate", type=int, default=2,
                       help="Run Hough recalibration every N frames")
    court.add_argument("--buffer-size", type=int, default=5, help="Frames in the smoothing buffer")

    vis = parser.add_argument_group("visualization")
    vis.add_argument("--map-orientation", choices=["horizontal", "vertical"], default="horizontal")
    vis.add_argument("--map-padding", type=int, default=200, help="BEV canvas padding in pixels")
    vis.add_argument("--combine", action=argparse.BooleanOptionalAction, default=True,
                     help="Show source and BEV views in one combined window")
    vis.add_argument("--debug-cols", type=int, default=3, help="Columns in the debug view grid")
    vis.add_argument("--debug-resolution", default="720p", help="Debug view resolution, e.g. 720p")
    vis.add_argument("--sleep", type=float, default=2.0, help="Delay between frames in seconds")

    repro = parser.add_argument_group("reproducibility")
    repro.add_argument("--seed", type=int, default=0)
    repro.add_argument("--deterministic", action="store_true",
                       help="Force deterministic cuDNN (reproducible but slower)")

    args = parser.parse_args(argv)

    if not is_stream(args.source) and not Path(args.source).exists():
        parser.error(f"--source not found: {args.source}")

    return args

def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit(f"Error: device '{name}' requested but CUDA is not available.")
    return device
    
def print_env_info(device: torch.device) -> None:
    info = f"PyTorch {torch.__version__} | device: {device}"
    if device.type == "cuda":
        info += f" ({torch.cuda.get_device_name(device)}, CUDA {torch.version.cuda})"
    print(info)

def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        source_path=args.source,
        use_wrapper=args.wrapper,
        enable_tracking=args.tracking,
        is_combine_result=args.combine,
        save_video=args.save_video,
        map_padding=args.map_padding,
        class_id_filter=None if args.all_classes else args.classes,
        is_manual_picker=args.manual_picker,
        frame_count=args.frame_count,
        model_path=args.model,
        sleep_second=args.sleep,
        map_orientation=args.map_orientation,
        is_run_detect=args.detect,
        recalibration_rate=args.recalibration_rate,
        buffer_size=args.buffer_size,
        debug_cols=args.debug_cols,
        debug_resolution=args.debug_resolution,
    )

    
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
    #                         map_pprint_vectoradding=5, 
    #                         class_id_filter=target_classes,
    #                         is_manual_picker=False,
    #                         frame_count=2, 
    #                         model_path="yolov8s.pt")
    # is_finished = tracker_pipeline_v2(config, device)

    target_source3 = volley_dir
    # target_classes = None
    target_classes = [0]
    check_path(target_source3)
    return PipelineConfig(source_path=str(target_source3),
                            use_wrapper=True,
                            enable_tracking=True,
                            is_combine_result=True, 
                            save_video=False,
                            map_padding=200, 
                            class_id_filter=target_classes,
                            is_manual_picker=False,
                            frame_count=80, 
                            model_path="yolov8s.pt",
                            sleep_second=2,
                            map_orientation="horizontal",
                            is_run_detect=True,
                            recalibration_rate=2,
                            buffer_size=5,
                            debug_cols=3,
                            debug_resolution="720p")

def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    print_env_info(device)

    config = build_config(args)
    try:
        ok = flow4(config, device)
    except KeyboardInterrupt:
        # print("\nInterrupted.")
        return 130

    print("Pipeline ran successfully." if ok else "Pipeline failed.")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
