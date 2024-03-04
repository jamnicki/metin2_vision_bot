import sys
from itertools import cycle
from pathlib import Path
from random import choice
from time import perf_counter, sleep
from typing import Tuple

import cv2
import numpy as np
import spacy
from loguru import logger
from torch import where as torch_where
from ultralytics import YOLO, checks

from game_controller import GameController, Key
from settings import CAP_MAX_FPS, MODELS_DIR, WINDOW_HEIGHT, GameBind, UserBind
from utils import setup_logger
from vision_detector import VisionDetector
from utils import channel_generator


def main(debug):
    checks()
    setup_logger(script_name=Path(__file__).name, level="INFO" if not debug else "DEBUG")
    run(debug)


def run(debug):
    yolo = YOLO(MODELS_DIR / "valium_idle_metiny_yolov8s.pt")
    logger.info("YOLO model loaded.")

    vision = VisionDetector()
    logger.info("Vision detector loaded.")

    game = GameController(vision_detector=vision, start_delay=2)
    logger.info("Game controller loaded.")

    channel_gen = channel_generator(1, 8)

    YOLO_CONFIDENCE_THRESHOLD = 0.8
    CHANNEL_TIMEOUT = 20
    LOOKING_AROUND_MOVE_CAMERA_PRESS_TIME = 0.4
    WALK_TO_METIN_TIME = 1.5

    # METIN_CLS = 0  # smierci sohan
    # METIN_DESTROY_TIME = 8  # smierci sohan | poly + masne eq + IS

    METIN_CLS = 1  # upadku polana
    METIN_DESTROY_TIME = 3  # upadku polana | poly + masne eq + IS

    game.calibrate_camera()
    game.move_camera_down(press_time=0.7)

    while game.is_running:
        channel = next(channel_gen)
        game.change_to_channel(channel)
        
        game.use_boosters()

        t0 = perf_counter()
        metin_detected = False
        timed_out = False
        while not metin_detected:
            frame = vision.capture_frame()
            if frame is None:
                game.restart_game()
                stage3_first_frame = True
                continue

            if vision.logged_out(frame):
                logger.warning("Logged out. Re-logging...")
                game.login()
                sleep(3)  # wait out relogging blockage
                game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                continue

            if vision.is_loading(frame=vision.capture_frame()):
                sleep(10)
                if vision.is_loading(frame=vision.capture_frame()):
                    logger.warning("Loading is taking too long (>10s), something is wrong. Escaping to logging menu...")
                    game.tap_key(Key.esc, press_time=2)
                    sleep(5)
                    continue
                continue

            frame_after_poly_det, polymorphed = vision.is_polymorphed(frame=vision.capture_frame())
            if not polymorphed:
                game.toggle_skill(UserBind.AURA, reset_animation=False)
                game.use_polymorph()

            if perf_counter() - t0 > CHANNEL_TIMEOUT:
                timed_out = True
                logger.warning(f"Metin not found. Switching to next channel...")
                break

            latest_frame = vision.capture_frame()
            yolo_results = yolo.predict(
                source=VisionDetector.fill_non_clickable_wth_black(latest_frame),
                conf=YOLO_CONFIDENCE_THRESHOLD,
                verbose=debug
            )[0]
            any_yolo_results = len(yolo_results.boxes.cls) > 0
            metins_idxs = torch_where(yolo_results.boxes.cls == METIN_CLS)
            metin_detected = metins_idxs[0].shape[0] > 0
            logger.error(f"{metins_idxs=} {metin_detected=}")
            if not metin_detected:
                game.move_camera_right(press_time=LOOKING_AROUND_MOVE_CAMERA_PRESS_TIME)
                logger.warning(f"Metin not found. Looking around, retrying...")
                continue
        
        if timed_out and not metin_detected:
            continue

        metins_xywh = yolo_results.boxes.xywh[metins_idxs]
        metins_distance_to_center = np.linalg.norm(metins_xywh[:, :2] - np.array(vision.center), axis=1)
        closest_metin_idx = metins_distance_to_center.argmin()
        closest_metin_bbox_xywh = yolo_results.boxes.xywh[closest_metin_idx]
        closest_metin_bbox_center = closest_metin_bbox_xywh[:2]
        closest_metin_center_global = vision.get_global_pos(closest_metin_bbox_center)

        game.click_at(closest_metin_center_global)

        sleep(WALK_TO_METIN_TIME)

        game.start_attack()
        game.idle(METIN_DESTROY_TIME, pickup=True)
        game.pickup_many(uses=2)
        game.stop_attack()

        butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
        if butelka_dywizji_filled:
            game.move_full_butelka_dywizji()
            game.use_next_butelka_dywizji()


if __name__ == '__main__':
    main(debug=True)
    logger.success("Bot terminated.")
