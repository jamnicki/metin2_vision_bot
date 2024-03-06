import click
import sys
from pathlib import Path
from random import choice
from time import perf_counter, sleep
from datetime import timedelta
from typing import Tuple
from warnings import filterwarnings

try:
    import pl_core_news_lg
except ImportError:
    import spacy
    spacy.cli.download("pl_core_news_lg")
    import pl_core_news_lg

import cv2
import numpy as np
from loguru import logger
from torch import where as torch_where
from ultralytics import YOLO
from ultralytics import checks as yolo_checks

from game_controller import GameController, Key
from settings import CAP_MAX_FPS, MODELS_DIR, WINDOW_HEIGHT, GameBind, UserBind
from utils import setup_logger
from vision_detector import VisionDetector
from utils import channel_generator


@click.command()
@click.option("--stage", default=0, type=int, show_default=True, help="Stage to start from.")
@click.option("--log-level",
              default="INFO",
              show_default=True,
              type=click.Choice(["TRACE", "DEBUG", "INFO"], case_sensitive=False),
              help="Set the logging level."
)
def main(stage, log_level):
    log_level = log_level.upper()
    setup_logger(script_name=Path(__file__).name, level=log_level)
    # q = input(
    #     "\nPlease ensure that:"
    #     "\n\t- the game is running in 800x600 resolution,"
    #     "\n\t- the game is in windowed mode,"
    #     "\n\t- the game is in the foreground,"
    #     "\n\t- camera view is set to 'closer' (Options > System Settings > Camera -> Further),"
    #     "\n\t- fog is turned off (Options > System Settings > Fog -> Light),"
    #     "\n\t- the character stands in front of the dungeon NPC (otherwise you will wait for timeout),"
    #     "\n\t- the character is not polymorphed,"
    #     "\n\t- the character is standing,"
    #     "\n\t- minimap is visible,"
    #     "\n\t- equipment window is closed,"
    #     "\n\t- chat messages are turned off"
    #     "\n\nPress [Enter] to continue or [q] to terminate...\n > "
    # )
    # if "q" in q.lower():
    #     logger.warning("Terminated by user.")
    #     exit()
    logger.warning("Starting the bot...")
    run(stage, log_level)


def run(stage, log_level):
    yolo_checks()
    yolo_verbose = log_level in ["TRACE", "DEBUG"]
    yolo = YOLO(MODELS_DIR / "valium_polana_yolov8s.pt", device="gpu")

    nlp = pl_core_news_lg.load()

    vision = VisionDetector()
    game = GameController(vision_detector=vision, start_delay=2)
    start_time = perf_counter()
    channel_gen = channel_generator(3, 6)

    game.hide_minimap()

    REENTER_WAIT = 4
    YOLO_CONFIDENCE_THRESHOLD = 0.8
    YOLO_METIN_CONFIDENCE_THRESHOLD = 0.85
    NONSENSE_MSG_SIMILARITY_THRESHOLD = 0
    STAGE_NAMES = ["before_enter", "stage_200_mobs", "stage_minibosses", "stage_metins", "stage_item_drop", "stage_boss"]
    STAGE_TIMEOUT = [
        90,       # before_enter
        60 * 3,   # stage_200_mobs
        60 * 5,   # stage_minibosses
        60 * 8,   # stage_metins
        60 * 5,   # stage_item_drop
        60 * 5,   # stage_boss
    ]
    WALK_TIME_TO_METIN = 10
    METIN_DESTROY_TIME = 25

    stage1_task_msg = "Pokonajcie 200 potworów."
    stage2_task_msg = "Pokonajcie wszystkie bossy."
    stage3_task_msg = "Zniszczcie wszystkie kamienie metin."
    stage4_task_msg = "Zdobądźcie Runo Lasu z potworów."
    stage5_task_msg = "Pokonajcie bossa."

    stage_enter_times = [-1] * 6
    stage_first_times = [True] * 6
    curr_fps = 0
    destroyed_metins = 0
    metin_detected = False
    stage3_first_frame = True
    while game.is_running:
        logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})")
        cap_t0 = perf_counter()

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

        if vision.is_loading(frame):
            sleep(10)
            if vision.is_loading(frame=vision.capture_frame()):
                logger.warning("Loading is taking too long (>10s), something is wrong. Escaping to logging menu...")
                game.tap_key(Key.esc, press_time=2)
                sleep(5)
                continue
            continue

        frame_contains_valium_msg = vision.frame_contains_valium_message(frame=vision.capture_frame())
        if frame_contains_valium_msg:
            butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
            if butelka_dywizji_filled:
                game.move_full_butelka_dywizji()
                game.use_next_butelka_dywizji()
            logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
            sleep(5)
            continue

        fps_text_pos = (2, vision.h - 100)
        frame = cv2.putText(frame, f"FPS: {curr_fps:.1f}", fps_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA, False)

        frame_after_poly_det, polymorphed = vision.is_polymorphed(frame)

        # latest_frame = vision.capture_frame()
        # yolo_results = yolo.predict(
        #     source=VisionDetector.fill_non_clickable_wth_black(latest_frame),
        #     conf=YOLO_CONFIDENCE_THRESHOLD,
        #     verbose=yolo_verbose
        # )[0]
        # any_yolo_results = len(yolo_results.boxes.cls) > 0
        # frame_wth_yolo_dets = yolo_results.plot(line_width=2, font_size=18)


        # =============================== START ===============================

        # ENTER DUNGEON

        dung_npc_cls = 1
        stage0_timed_out = False
        if STAGE_NAMES[stage] == "before_enter":
            if stage_first_times[stage]:
                stage_first_times[stage] = False
                stage_enter_times[stage] = perf_counter()
                game.unmount()
                game.calibrate_camera()
                game.zoomin_camera(press_time=0.3)
                game.move_camera_down(press_time=0.8)
                sleep(1)  # wait for the camera to stabilize before capturing the frame

            latest_frame = vision.capture_frame()
            yolo_results = yolo.predict(
                source=VisionDetector.fill_non_clickable_wth_black(latest_frame),
                conf=YOLO_CONFIDENCE_THRESHOLD,
                verbose=yolo_verbose
            )[0]
            any_yolo_results = len(yolo_results.boxes.cls) > 0

            if stage_enter_times[stage] != -1 and perf_counter() - stage_enter_times[stage] > STAGE_TIMEOUT[stage]:
                game.tap_key(Key.enter)  # in case the npc dialog is open
                sleep(2)
                game.tap_key(Key.enter)
                sleep(5)
                stage1_timed_out = True
                logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_TIMEOUT[stage]}s). Teleporting to NPC and starting over...")
                game.teleport_to_polana()
                sleep(3)  # wait out relogging blockage
                game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                stage = 0
                stage_enter_times = [-1] * 6
                stage_first_times = [True] * 6
                destroyed_metins = 0
                metin_detected = False
                stage3_first_frame = True
                continue

            if not any_yolo_results or dung_npc_cls not in yolo_results.boxes.cls:
                logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Dungeon NPC not found. Retrying...")
                game.move_camera_left(press_time=0.6)
                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame=vision.capture_frame())
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue

                continue

            logger.debug(f"{yolo_results.boxes.cls=}")
            dung_npc_bbox_idx = torch_where(yolo_results.boxes.cls == dung_npc_cls)[0][0].item()

            logger.debug(f"{yolo_results.boxes[dung_npc_bbox_idx]=}")
            dung_npc_xywh = yolo_results.boxes.xywh[dung_npc_bbox_idx]
            dung_npc_center = dung_npc_xywh[:2]
            dung_npc_center_global = vision.get_global_pos(dung_npc_center)

            game.click_at(dung_npc_center_global)
            sleep(2)
            game.tap_key(Key.enter)
            sleep(2)
            game.tap_key(Key.enter)
            sleep(25)  # wait for the stage1 task message to appear

            dung_message = VisionDetector.get_dungeon_message(frame=vision.capture_frame())
            msg_similarity = nlp(dung_message).similarity(nlp(stage1_task_msg))
            first_stage_task_msg_visible = msg_similarity > 0.6
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {dung_message=} {msg_similarity=:.3f}")
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {first_stage_task_msg_visible=} {stage1_task_msg=}")
            if first_stage_task_msg_visible:
                stage = 1
                logger.success("Dungeon entered. Starting the stages sequence.")
                continue

        # STAGE 1

        if STAGE_NAMES[stage] == "stage_200_mobs":
            if stage_first_times[stage]:
                game.polymorph_off()
                game.mount()
                game.calibrate_camera()

            stage1_all_mobs_killed = False
            stage1_timed_out = False
            while not stage1_all_mobs_killed:
                if stage_enter_times[stage] != -1 and perf_counter() - stage_enter_times[stage] > STAGE_TIMEOUT[stage]:
                    stage1_timed_out = True
                    logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_TIMEOUT[stage]}s). Teleporting to NPC and starting over...")
                    game.teleport_to_polana()
                    sleep(3)  # wait out relogging blockage
                    game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                    stage = 0
                    stage_enter_times = [-1] * 6
                    stage_first_times = [True] * 6
                    destroyed_metins = 0
                    metin_detected = False
                    stage3_first_frame = True
                    break

                # attack for 20 seconds while mounted
                game.use_boosters()
                game.start_attack()
                game.lure_many()
                game.idle(time=7, lure=True, pickup=True, turn_randomly=True)
                game.move_camera_left(press_time=0.7)
                game.stop_attack()
                game.pickup()

                # steer randomly for 2 seconds (escape the ghost mobs that are attacking)
                game.press_key(GameBind.CIECIE_Z_SIODLA)
                game.steer_randomly(press_time=2)
                game.release_key(GameBind.CIECIE_Z_SIODLA)

                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame=vision.capture_frame())
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue

                dung_message = VisionDetector.get_dungeon_message(frame=vision.capture_frame())
                msg_similarity = nlp(dung_message).similarity(nlp(stage2_task_msg))
                next_task_msg = msg_similarity > 0.75
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {dung_message=} {msg_similarity=:.3f}")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {next_task_msg=} {stage2_task_msg=}")
                if next_task_msg:
                    stage1_all_mobs_killed = True
                    game.pickup_many()
                    game.stop_attack()
                    game.unmount()
                    stage = 2
                    logger.success(f"Stage 'stage_200_mobs' (1) completed.")
                    sleep(2)  # wait for the next stage to load
                    break

                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_first_times=}")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_enter_times=}")
                if stage_first_times[stage] and stage_enter_times[stage] == -1:
                    stage_first_times[stage] = False
                    stage_enter_times[stage] = perf_counter()

                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | Not all mobs killed. Keep attacking...")
                continue

            game.pickup_many()
            game.stop_attack()
            game.unmount()

            if stage1_timed_out:
                continue


        if STAGE_NAMES[stage] == "stage_minibosses":
            # - zabij 3 bossy
            #     - lure
            #     - use passive skills
            #     - use polymorph
            #     - atakuj
            #     - random turn
            #     - pickup
            #     aż do wykrycia następneg komunikatu

            if stage_first_times[stage]:
                game.unmount()
                game.use_polymorph()

            stage2_all_minibosses_killed = False
            stage2_timed_out = False
            while not stage2_all_minibosses_killed:
                if stage_enter_times[stage] != -1 and perf_counter() - stage_enter_times[stage] > STAGE_TIMEOUT[stage]:
                    stage2_timed_out = True
                    logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_TIMEOUT[stage]}s). Teleporting to NPC and starting over...")
                    game.teleport_to_polana()
                    sleep(3)  # wait out relogging blockage
                    game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                    stage = 0
                    stage_enter_times = [-1] * 6
                    stage_first_times = [True] * 6
                    destroyed_metins = 0
                    metin_detected = False
                    stage3_first_frame = True
                    break

                game.start_attack()
                game.lure()
                game.pickup()
                game.turn_randomly()

                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame=vision.capture_frame())
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue
                
                dung_message = VisionDetector.get_dungeon_message(frame=vision.capture_frame())
                msg_similarity = nlp(dung_message).similarity(nlp(stage3_task_msg))
                next_task_msg = msg_similarity > 0.85
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {dung_message=} ({msg_similarity=:.3f})")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {next_task_msg=} {stage3_task_msg=}")
                if next_task_msg:
                    stage2_all_minibosses_killed = True
                    break


                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_first_times=}")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_enter_times=}")
                if stage_first_times[stage] and stage_enter_times[stage] == -1:
                    stage_first_times[stage] = False
                    stage_enter_times[stage] = perf_counter()

            if stage2_timed_out:
                continue

            game.stop_attack()

            logger.success("Stage 'stage_minibosses' (2) completed.")
            stage = 3
            sleep(10)  # wait for the next stage to load
            continue


        metin_cls = 0
        if STAGE_NAMES[stage] == "stage_metins":
            # - zniszcz 5 metinów
            #     - kamera na skos wykrywanie
            #     - obrót w lewo; next frame; sprawdz czy metin TAK: atakuj NIe: repeat
            #     - naciśnij na metina;
            #     - sleep(1)
            #     - atakuj dopóki okno celu nie zniknie (ocr -> czy nazwa metina jest w stringu?)
            #     - zniszczone metiny += 1 ; aż do skutku (5 metinów)
            #     - sleep(1)
            #     - sprawdz czy komunikat z stage o metinach zniknął TAK: git, lecim dalej NIE: kurwa nie wiem

            dung_message = VisionDetector.get_dungeon_message(frame=vision.capture_frame())
            msg_similarity = nlp(dung_message).similarity(nlp(stage4_task_msg))
            next_task_msg = msg_similarity > 0.8
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {dung_message=} ({msg_similarity=:.3f})")
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {next_task_msg=} {stage4_task_msg=}")
            if next_task_msg:
                game.polymorph_off()
                game.pickup_many()

                logger.success("Stage 'stage_metins' (3) completed.")
                stage = 4
                sleep(4)  # wait for the next stage to load
                continue

            if not stage_first_times[stage] and stage_enter_times[stage] != -1 and perf_counter() - stage_enter_times[stage] > STAGE_TIMEOUT[stage]:
                logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_TIMEOUT[stage]}s). Teleporting to NPC and starting over...")
                game.teleport_to_polana()
                sleep(3)  # wait out relogging blockage
                game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                stage = 0
                stage_enter_times = [-1] * 6
                stage_first_times = [True] * 6
                destroyed_metins = 0
                metin_detected = False
                stage3_first_frame = True
                continue

            if stage_first_times[stage]:
                obstacle_avoided = False
                stage_first_times[stage] = False
                stage_enter_times[stage] = perf_counter()
                game.unmount()
                game.calibrate_camera()
                game.zoomin_camera(press_time=0.1)
                game.move_camera_down(press_time=0.8)

            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_first_times=}")
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_enter_times=}")

            latest_frame = vision.capture_frame()
            yolo_results = yolo.predict(
                source=VisionDetector.fill_non_clickable_wth_black(latest_frame),
                conf=YOLO_METIN_CONFIDENCE_THRESHOLD,
                verbose=yolo_verbose
            )[0]
            any_yolo_results = len(yolo_results.boxes.cls) > 0
            if not any_yolo_results:
                game.move_camera_right(press_time=0.2)
                logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Metin not found. Looking around, retrying...")
                continue

            if not obstacle_avoided and destroyed_metins == 0:
                # avoid the obstacle
                game.steer_up_right(press_time=1)
                game.move_camera_left(press_time=0.5)
                obstacle_avoided = True
                continue

            metins_idxs = torch_where(yolo_results.boxes.cls == metin_cls)
            metin_detected = metins_idxs[0].shape[0] > 0
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {metins_idxs=} {metin_detected=}")
            if not metin_detected:
                game.move_camera_left(press_time=0.4)
                logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Metin not found. Looking around, retrying...")
                continue

            metins_xywh = yolo_results.boxes.xywh[metins_idxs]
            metins_distance_to_center = np.linalg.norm(metins_xywh[:, :2] - np.array(vision.center), axis=1)
            closest_metin_idx = metins_distance_to_center.argmin()
            closest_metin_bbox_xywh = yolo_results.boxes.xywh[closest_metin_idx]
            closest_metin_bbox_center = closest_metin_bbox_xywh[:2]
            closest_metin_center_global = vision.get_global_pos(closest_metin_bbox_center)

            game.use_polymorph()
            game.use_boosters()

            game.click_at(closest_metin_center_global)

            sleep(WALK_TIME_TO_METIN)

            game.start_attack()
            game.idle(time=METIN_DESTROY_TIME, pickup=True, use_boosters=True)
            game.stop_attack()

            destroyed_metins += 1
            metin_detected = False
            game.pickup_many()
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {destroyed_metins=}")


        if STAGE_NAMES[stage] == "stage_item_drop":
            # - znajdz liscie i użyj je
            #     - lure
            #     - mount; start_attack
            #     - pickup
            #     - czy Runo Leśne pojawiło się w eq? TAK: kliknij na nie ppm NIE: atakuj dalej 

            game.polymorph_off()
            game.mount()
            game.calibrate_camera()

            looking_for_item_t0 = perf_counter()
            next_stage_act_item_found = False
            stage4_timed_out = False
            while not next_stage_act_item_found:
                if stage_enter_times[stage] != -1 and perf_counter() - stage_enter_times[stage] > STAGE_TIMEOUT[stage]:
                    stage4_timed_out = True
                    logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_TIMEOUT[stage]}s). Teleporting to NPC and starting over...")
                    game.teleport_to_polana()
                    sleep(3)  # wait out relogging blockage
                    game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                    stage = 0
                    stage_enter_times = [-1] * 6
                    stage_first_times = [True] * 6
                    destroyed_metins = 0
                    metin_detected = False
                    stage3_first_frame = True
                    break

                game.show_eq_slot(1)

                # attack for 20 seconds while mounted
                game.use_boosters()
                game.start_attack()
                game.lure_many()
                game.idle(time=5, lure=True, pickup=True, turn_randomly=True)
                game.move_camera_left(press_time=0.7)
                game.stop_attack()
                game.pickup_many()

                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame=vision.capture_frame()):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame=vision.capture_frame())
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue

                item_dropped, item_dropped_conf, item_dropped_loc = vision.detect_runo_lesne_dropped(frame=vision.capture_frame())
                if item_dropped:
                    # so pick it up
                    game.stop_attack()
                    item_dropped_global_loc = vision.get_global_pos(item_dropped_loc)
                    game.press_key(GameBind.CIECIE_Z_SIODLA)
                    game.click_at(item_dropped_global_loc)
                    game.release_key(GameBind.CIECIE_Z_SIODLA)
                    game.pickup_many(uses=5)
                else:
                    # steer randomly for 2 seconds (escape the ghost mobs that are attacking)
                    game.press_key(GameBind.CIECIE_Z_SIODLA)
                    game.steer_randomly(press_time=2)
                    game.release_key(GameBind.CIECIE_Z_SIODLA)

                next_stage_act_item_found, item_found_conf, item_found_loc = vision.detect_runo_lesne(frame=vision.capture_frame())

                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_first_times=}")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_enter_times=}")
                if stage_first_times[stage] and stage_enter_times[stage] == -1:
                    stage_first_times[stage] = False
                    stage_enter_times[stage] = perf_counter()

                logger.info(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Runo Leśne not found. Retrying...")

            if stage4_timed_out:
                continue

            next_stage_item_global_loc = vision.get_global_pos(item_found_loc)
            logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | Next stage item found\t{item_found_conf=:.2f} {item_found_loc=}")

            sleep(1)
            game.click_at(next_stage_item_global_loc, right=True)
            game.stop_attack()
            game.unmount()
            game.hide_eq()

            logger.success("Stage 'stage_item_drop' (4) completed.")
            stage = 5
            sleep(4)  # wait for the next stage to load


        stage5_task_msg = "Pokonajcie bossa."
        double_boss_event_task_msg = "Pojawił się następny boss!"
        if STAGE_NAMES[stage] == "stage_boss":
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
            stage5_timed_out = False
            while not stage5_boss_killed:
                if stage_enter_times[stage] != -1 and perf_counter() - stage_enter_times[stage] > STAGE_TIMEOUT[stage]:
                    stage5_timed_out = True
                    logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_TIMEOUT[stage]}s). Teleporting to NPC and starting over...")
                    game.teleport_to_polana()
                    sleep(3)  # wait out relogging blockage
                    game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                    stage = 0
                    stage_enter_times = [-1] * 6
                    stage_first_times = [True] * 6
                    destroyed_metins = 0
                    metin_detected = False
                    stage3_first_frame = True
                    break

                frame = vision.capture_frame()
                if frame is None:
                    game.restart_game()
                    continue

                if vision.logged_out(frame):
                    logger.warning("Logged out. Re-logging...")
                    game.login()
                    continue

                frame_contains_valium_msg = vision.frame_contains_valium_message(frame=vision.capture_frame())
                if frame_contains_valium_msg:
                    butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
                    if butelka_dywizji_filled:
                        game.move_full_butelka_dywizji()
                        game.use_next_butelka_dywizji()
                    logger.warning("Valium message detected. Skipping this frame, recapturing in 5s...")
                    sleep(5)
                    continue
                
                game.lure()
                game.turn_randomly()
                dung_message = VisionDetector.get_dungeon_message(frame=vision.capture_frame())
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  | {dung_message=}")
                msg_similarity = nlp(dung_message).similarity(nlp(stage5_task_msg))
                msg_double_boss_event_similarity = nlp(dung_message).similarity(nlp(double_boss_event_task_msg))
                not_task_msg = NONSENSE_MSG_SIMILARITY_THRESHOLD < msg_similarity < 0.6
                double_boss_event_msg = msg_double_boss_event_similarity > 0.6
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {dung_message=} {msg_similarity=:.3f} {msg_double_boss_event_similarity=:.3f}")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {not_task_msg=} {double_boss_event_msg=}")
                if not_task_msg and not double_boss_event_msg:
                    stage5_boss_killed = True
                    break

                if double_boss_event_msg and perf_counter() - stage5_t0 > STAGE_TIMEOUT[stage]:
                    logger.warning(f"Stage {STAGE_NAMES[stage]} ({stage})  |  Timeout ({STAGE_5_TIMEOUT}s). Re-entering in {REENTER_WAIT}s...")
                    game.pickup_many(uses=5)
                    stage5_took_too_long = True
                    break

                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_first_times=}")
                logger.debug(f"Stage {STAGE_NAMES[stage]} ({stage})  |  {stage_enter_times=}")
                if stage_first_times[stage] and stage_enter_times[stage] == -1:
                    stage_first_times[stage] = False
                    stage_enter_times[stage] = perf_counter()

            if stage5_timed_out:
                continue

            game.pickup_many(uses=3)
            game.stop_attack()
            game.polymorph_off()

            # because of last stage completed
            stage_completion_times = [perf_counter() - t for t in stage_enter_times]
            if not stage5_took_too_long:
                logger.success(f"Boss has been killed! Dungeon completed. Re-entering in {REENTER_WAIT}s...")
                for i, stage_name in enumerate(STAGE_NAMES):
                    logger.success(f"({i}) {stage_name:>20}: {timedelta(seconds=stage_completion_times[i])}")
                logger.success("")
                logger.success(f"Total: {timedelta(seconds=sum(stage_completion_times))}")

            game.calibrate_camera()
            game.tap_key(GameBind.MOVE_RIGHT, press_time=0.3)
            game.tap_key(GameBind.MOVE_FORWARD, press_time=0.1)
            sleep(REENTER_WAIT)

            # reset the parameters
            stage = 0
            stage_enter_times = [-1] * 6
            # stage_first_times[1:] = [True] * 5  # keep stage0 as False for continuous re-entering
            stage_first_times = [True] * 6
            destroyed_metins = 0
            metin_detected = False
            stage3_first_frame = True
            continue


    if game.is_running:
        game.exit()
        exit()


if __name__ == '__main__':
    main()
    logger.success("Bot terminated.")
