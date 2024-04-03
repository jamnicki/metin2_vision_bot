import sys
from datetime import timedelta
from pathlib import Path
from random import choice
from time import perf_counter, sleep
from typing import Tuple
from warnings import filterwarnings

import click
import cv2
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from torch import where as torch_where
from ultralytics import YOLO
from ultralytics import checks as yolo_checks

from game_controller import GameController, Key
from settings import CAP_MAX_FPS, MODELS_DIR, WINDOW_HEIGHT, GameBind, UserBind
from utils import channel_generator, setup_logger
from vision_detector import VisionDetector


@click.command()
@click.option(
    "--stage", default=0, type=int, show_default=True, help="Stage to start from."
)
@click.option(
    "--log-level",
    default="TRACE",
    show_default=True,
    type=click.Choice(["TRACE", "DEBUG", "INFO"], case_sensitive=False),
    help="Set the logging level.",
)
@click.option(
    "--saved_credentials_idx",
    default=1,
    type=int,
    show_default=True,
    help="Saved credentials index to use.",
)
def main(stage, log_level, saved_credentials_idx):
    log_level = log_level.upper()
    setup_logger(script_name=Path(__file__).name, level=log_level)
    logger.warning("Starting the bot...")
    run(stage, log_level, saved_credentials_idx)


def run(stage, log_level, saved_credentials_idx):
    yolo_checks()
    yolo_verbose = log_level in ["TRACE", "DEBUG"]
    yolo = YOLO(MODELS_DIR / "global_fishbot_yolov8s.pt").to("cuda:0")

    vision = VisionDetector()
    game = GameController(vision_detector=vision, start_delay=2)

    YOLO_CONFIDENCE_THRESHOLD = 0.95
    FISH_CLS = 0

    while game.is_running:
        frame = vision.capture_frame()

        fishing_window = frame[77:304, 101:379]

        yolo_results = yolo.predict(
            source=fishing_window, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=yolo_verbose
        )[0]
        fish_detected = len(yolo_results.boxes.cls) > 0
        logger.debug(f"{fish_detected=}")

        if not fish_detected:
            logger.info("No fish detected.")
            sleep(0.25)
            continue

        fish_bbox_xywh = yolo_results.boxes.xywh[0]
        fish_bbox_center = fish_bbox_xywh[:2]

        fish_bbox_center_fixed = fish_bbox_center + np.array([101, 77])

        fish_bbox_center_global = vision.get_global_pos(fish_bbox_center_fixed)

        game.catch_fish(fish_bbox_center_global)

        sleep(1)

    if game.is_running:
        game.exit()
        exit()


if __name__ == "__main__":
    main()
    logger.success("Bot terminated.")
