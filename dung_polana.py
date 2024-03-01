import sys
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
from utils import set_logger_level
from vision_detector import VisionDetector


def get_first_pixel(img):
    return img[0, 0]

def scale_img(img, scale=5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def binarized_to_grayscale(binarized_img):
    # return np.array([[[x, x, x] for x in row] for row in binarized_img], dtype=np.uint8) * 255

    # Ensure the input is a NumPy array for efficient computation
    binarized_img = np.array(binarized_img, dtype=np.uint8)
    # Use broadcasting to replicate the binarized image across the RGB channels
    grayscale_img = np.expand_dims(binarized_img, axis=-1) * 255
    # Stack the single channel image across the three channels (RGB)
    return np.repeat(grayscale_img, 3, axis=-1)


def _before_next_frame(game, vision, frame, cap_t0):
    # vision.show_preview(vision.scale_frame(frame, scale=0.8))
    game.pickup()
    game.use_boosters()

    # compute_time = perf_counter() - cap_t0
    # curr_fps = 1 / compute_time
    # logger.debug(f"FPS: {curr_fps:<2.2f}")

    # cap_wait = max(0, 1 / CAP_MAX_FPS - compute_time)
    # sleep(cap_wait)


def main(debug):
    checks()
    set_logger_level(script_name=Path(__file__).name, level="INFO" if not debug else "DEBUG")
    run(debug)


def run(debug):
    yolo = YOLO(MODELS_DIR / "valium_polana_yolov8s.pt")
    nlp = spacy.load('pl_core_news_sm')

    vision = VisionDetector()
    game = GameController(vision_detector=vision, start_delay=5)
    start_time = perf_counter()

    game.hide_minimap()

    stage_names = ["before_enter", "stage_200_mobs", "stage_3_minibosses", "stage_5_metins", "stage_item_drop", "stage_boss"]
    REENTER_WAIT = 4
    YOLO_CONFIDENCE_THRESHOLD = 0.7
    NONSENSE_MSG_SIMILARITY_THRESHOLD = 0
    STAGE_5_TIMEOUT = 60

    stage = 0
    curr_fps = 0
    destroyed_metins = 0
    metin_detected = False
    stage3_first_frame = True
    while game.is_running:
        logger.debug(f"Stage: {stage_names[stage]}")
        cap_t0 = perf_counter()

        frame = vision.capture_frame()
        if frame is None:
            game.restart_game()
            stage3_first_frame = True
            continue

        if vision.logged_out(frame):
            logger.warning("Logged out. Re-logging...")
            game.login()
            continue

        frame_contains_valium_msg = vision.frame_contains_valium_message(frame)
        if frame_contains_valium_msg:
            butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame)
            if butelka_dywizji_filled:
                game.move_full_butelka_dywizji()
                game.use_next_butelka_dywizji()
            logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
            sleep(5)
            continue

        fps_text_pos = (2, vision.h - 100)
        frame = cv2.putText(frame, f"FPS: {curr_fps:.1f}", fps_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA, False)

        frame_after_poly_det, polymorphed = vision.is_polymorphed(frame)

        yolo_results = yolo.predict(
            source=VisionDetector.fill_non_clickable_wth_black(frame_after_poly_det),
            conf=YOLO_CONFIDENCE_THRESHOLD,
            verbose=debug
        )[0]
        any_yolo_results = len(yolo_results.boxes.cls) > 0
        frame_wth_yolo_dets = yolo_results.plot(line_width=2, font_size=18)


        # =============================== START ===============================

        # ENTER DUNGEON

        dung_npc_cls = 1
        if stage == 0:
            if not any_yolo_results or dung_npc_cls not in yolo_results.boxes.cls:
                logger.warning(f"Stage {stage}  |  Dungeon NPC not found. Retrying...")
                game.move_camera_left(press_time=0.1)
                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame)
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame)
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue

                yolo_results = yolo.predict(
                    source=VisionDetector.fill_non_clickable_wth_black(frame),
                    conf=YOLO_CONFIDENCE_THRESHOLD,
                    verbose=debug
                )[0]
                _before_next_frame(game, vision, frame, cap_t0)
                continue

            logger.warning(yolo_results.boxes.cls)
            dung_npc_bbox_idx = torch_where(yolo_results.boxes.cls == dung_npc_cls)[0][0].item()

            logger.warning(yolo_results.boxes[dung_npc_bbox_idx])
            dung_npc_xywh = yolo_results.boxes.xywh[dung_npc_bbox_idx]
            dung_npc_center = dung_npc_xywh[:2]
            dung_npc_center_global = vision.get_global_pos(dung_npc_center)

            game.click_at(dung_npc_center_global)
            sleep(0.8)
            game.tap_key(Key.enter)
            sleep(0.8)
            game.tap_key(Key.enter)
            sleep(2)  # wait for the dungeon to load

            stage = 1
            logger.info("Dungeon entered. Starting the stages sequence.")
            game.calibrate_camera()

        # # STAGE 1

        stage1_task_msg = "Pokonajcie 200 potworów."
        if stage == 1:
            stage1_all_mobs_killed = False
            while not stage1_all_mobs_killed:
                game.unmount()
                game.lure_many
                game.start_attack()
                game.pickup()

                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame)
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame)
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue

                game.lure_many(uses=3)
                game.pickup_many()
                game.turn_randomly()

                dung_message = VisionDetector.get_dungeon_message(frame)
                msg_similarity = nlp(dung_message).similarity(nlp(stage1_task_msg))
                not_task_msg = NONSENSE_MSG_SIMILARITY_THRESHOLD < msg_similarity < 0.6
                if not_task_msg:
                    logger.debug(f"Stage {stage}  | {dung_message=} {msg_similarity=:.3f}")
                    stage1_all_mobs_killed = True
                    break

                logger.debug(f"Stage {stage}  | Not all mobs killed. Keep attacking...")
                _before_next_frame(game, vision, frame_wth_yolo_dets, cap_t0)
                continue

            game.pickup_many()
            game.stop_attack()
            # game.unmount()

            stage = 2
            logger.error("Stage 1 completed.")
            sleep(2)  # wait for the next stage to load


        stage2_task_msg = "Pokonajcie wszystkie bossy."
        if stage == 2:
            # - zabij 3 bossy
            #     - lure
            #     - use passive skills
            #     - use polymorph
            #     - atakuj
            #     - random turn
            #     - pickup
            #     aż do wykrycia następneg komunikatu

            game.lure_many()
            game.toggle_passive_skills()

            stage1_all_minibosses_killed = False
            while not stage1_all_minibosses_killed:
                game.use_polymorph()
                game.start_attack()
                game.lure()
                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame)
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame)
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue
                
                dung_message = VisionDetector.get_dungeon_message(frame)
                msg_similarity = nlp(dung_message).similarity(nlp(stage2_task_msg))
                not_task_msg = NONSENSE_MSG_SIMILARITY_THRESHOLD < msg_similarity < 0.6
                if not_task_msg:
                    logger.debug(f"Stage {stage}  | {dung_message=} ({msg_similarity=:.3f})")
                    stage1_all_minibosses_killed = True
                    _before_next_frame(game, vision, frame, cap_t0)
                    break

                game.turn_randomly()
                game.pickup()
                game.lure()
                game.lure()

            game.stop_attack()
            game.polymorph_off()

            logger.info("Stage 2 completed.")
            stage = 3
            sleep(4)  # wait for the next stage to load


        metin_cls = 0
        stage3_task_msg = "Zniszcz 5 kamieni metin."
        if stage == 3:
            # - zniszcz 5 metinów
            #     - kamera na skos wykrywanie
            #     - obrót w lewo; next frame; sprawdz czy metin TAK: atakuj NIe: repeat
            #     - naciśnij na metina;
            #     - sleep(1)
            #     - atakuj dopóki okno celu nie zniknie (ocr -> czy nazwa metina jest w stringu?)
            #     - zniszczone metiny += 1 ; aż do skutku (5 metinów)
            #     - sleep(1)
            #     - sprawdz czy komunikat z stage o metinach zniknął TAK: git, lecim dalej NIE: kurwa nie wiem

            if stage3_first_frame:
                game.calibrate_camera()
                game.move_camera_down(press_time=1.5)
                stage3_first_frame = False

            if not any_yolo_results:
                if destroyed_metins == 0:
                    game.move_camera_right(press_time=0.2)
                else:
                    game.move_camera_left(press_time=0.2)
                logger.warning(f"Stage {stage}  |  Metin not found. Looking around, retrying...")
                _before_next_frame(game, vision, frame, cap_t0)
                continue

            metins_idxs = torch_where(yolo_results.boxes.cls == metin_cls)
            metin_detected = metins_idxs[0].shape[0] > 0
            logger.error(f"Stage {stage}  |  {metins_idxs=} {metin_detected=}")
            if not metin_detected:
                game.move_camera_right(press_time=0.5)
                logger.warning(f"Stage {stage}  |  Metin not found. Looking around, retrying...")
                _before_next_frame(game, vision, frame, cap_t0)
                continue

            if destroyed_metins < 5:
                metins_xywh = yolo_results.boxes.xywh[metins_idxs]
                metins_distance_to_center = np.linalg.norm(metins_xywh[:, :2] - np.array(vision.center), axis=1)
                closest_metin_idx = metins_distance_to_center.argmin()
                closest_metin_bbox_xywh = yolo_results.boxes.xywh[closest_metin_idx]
                closest_metin_bbox_center = closest_metin_bbox_xywh[:2]
                closest_metin_center_global = vision.get_global_pos(closest_metin_bbox_center)

                game.use_polymorph()

                game.click_at(closest_metin_center_global)
                if destroyed_metins == 0:
                    game.tap_key(GameBind.MOVE_RIGHT, press_time=0.8)
                    
                walk_to_metin_time = 6
                sleep(walk_to_metin_time)

                # metin_destroy_time = 24  # poly + masne eq
                # metin_destroy_time = 47  # mounted + masne eq
                # metin_destroy_time = 25  # mounted + masne eq + IS
                metin_destroy_time = 15  # poly + masne eq + IS
                sleep(metin_destroy_time)

                if destroyed_metins == 0:
                    game.move_camera_left(press_time=2.6)

                destroyed_metins += 1
                metin_detected = False
                game.pickup_many()
                logger.warning(f"Stage {stage}  |  {destroyed_metins=}")
                _before_next_frame(game, vision, frame, cap_t0)

            dung_message = VisionDetector.get_dungeon_message(frame)
            logger.debug(f"Stage {stage}  | {dung_message=}")
            msg_similarity = nlp(dung_message).similarity(nlp(stage5_task_msg))
            task_msg_changed = msg_similarity < 0.6
            if destroyed_metins >= 5 and task_msg_changed:
                game.polymorph_off()
                game.pickup_many()
                _before_next_frame(game, vision, frame, cap_t0)

                logger.info("Stage 3 completed.")
                stage = 4
                sleep(4)  # wait for the next stage to load


        if stage == 4:
            # - znajdz liscie i użyj je
            #     - lure
            #     - mount; start_attack
            #     - pickup
            #     - czy Runo Leśne pojawiło się w eq? TAK: kliknij na nie ppm NIE: atakuj dalej 

            game.calibrate_camera()
            game.mount()

            looking_for_item_t0 = perf_counter()
            next_stage_act_item_found = False
            while not next_stage_act_item_found:
                game.show_eq()
                game.start_attack()
                game.lure()
                game.pickup_many(uses=3)

                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame)
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame)
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(8)
                    continue

                item_dropped, item_dropped_conf, item_dropped_loc = vision.detect_runo_lesne_dropped(frame)
                if item_dropped:
                    # so pick it up
                    game.stop_attack()
                    game.unmount()
                    sleep(1)
                    item_dropped_global_loc = vision.get_global_pos(item_dropped_loc)
                    game.click_at(item_dropped_global_loc)
                    sleep(1)
                    game.tap_key(UserBind.WIREK, press_time=1.3)
                    game.pickup_many(uses=5)

                next_stage_act_item_found, item_found_conf, item_found_loc = vision.detect_runo_lesne(frame)

                _before_next_frame(game, vision, frame, cap_t0)
                logger.error(f"Stage {stage}  |  Runo Leśne not found. Retrying...")

            next_stage_item_global_loc = vision.get_global_pos(item_found_loc)
            logger.debug(f"Stage {stage}  | Next stage item found\t{item_found_conf=:.2f} {item_found_loc=}")

            sleep(1)
            game.click_at(next_stage_item_global_loc, right=True)
            game.stop_attack()
            game.unmount()
            game.hide_eq()

            logger.info("Stage 4 completed.")
            stage = 5
            sleep(4)  # wait for the next stage to load


        stage5_task_msg = "Pokonajcie bossa."
        double_boss_event_task_msg = "Pojawił się następny boss!"
        if stage == 5:
            # - zabij kapitana
            #     - lure
            #     - atakuj w miejscu + obrót
            #     - pickup_many(uses=5?)
            #     - atakuj aż do braku komunikatu "Zabijcie bossa." ?

            game.lure_many()
            game.toggle_passive_skills()
            game.use_polymorph()
            game.start_attack()

            stage5_t0 = perf_counter()
            stage5_boss_killed = False
            stage5_took_too_long = False
            while not stage5_boss_killed:
                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame)
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame)
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue
                
                game.lure()
                game.turn_randomly()
                dung_message = VisionDetector.get_dungeon_message(frame)
                logger.debug(f"Stage {stage}  | {dung_message=}")
                msg_similarity = nlp(dung_message).similarity(nlp(stage5_task_msg))
                msg_double_boss_event_similarity = nlp(dung_message).similarity(nlp(double_boss_event_task_msg))
                not_task_msg = NONSENSE_MSG_SIMILARITY_THRESHOLD < msg_similarity < 0.6
                double_boss_event_msg = msg_double_boss_event_similarity > 0.6
                if not_task_msg and not double_boss_event_msg:
                    logger.debug(f"Stage {stage}  |  {dung_message=} {msg_similarity=:.3f} {msg_double_boss_event_similarity=:.3f}")
                    stage5_boss_killed = True
                    break

                if double_boss_event_msg and perf_counter() - stage5_t0 > STAGE_5_TIMEOUT:
                    logger.warning(f"Stage {stage}  |  Timeout ({STAGE_5_TIMEOUT}s). Re-entering in {REENTER_WAIT}s...")
                    game.pickup_many(uses=5)
                    stage5_took_too_long = True
                    break

            game.pickup_many(uses=3)
            game.stop_attack()
            game.polymorph_off()

            # because of last stage completed
            if not stage5_took_too_long:
                logger.info(f"Boss has been killed! Dungeon completed. Re-entering in {REENTER_WAIT}s...")
            sleep(REENTER_WAIT)

            stage = 0
            looking_around = False
            destroyed_metins = 0
            metin_detected = False
            stage3_first_frame = True
            
            game.calibrate_camera()
            game.tap_key(GameBind.MOVE_RIGHT, press_time=0.3)
            game.tap_key(GameBind.MOVE_FORWARD, press_time=0.1)


        # every tick
        _before_next_frame(game, vision, frame_wth_yolo_dets, cap_t0)

    if game.is_running:
        game.exit()
        exit()


if __name__ == '__main__':
    main(debug=True)
    logger.success("Bot terminated.")
