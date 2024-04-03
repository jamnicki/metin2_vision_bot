import os
import re
from random import choice, choices, uniform
from time import perf_counter, sleep
from typing import Generator, Optional, Tuple

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR/tesseract.exe"

import cv2
import numpy as np
import win32gui
from loguru import logger
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.keyboard import Listener as KeyboardListener
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
from pynput.mouse import Listener as MouseListener

import positions
from settings import (
    GAME_VIEW_POLY_OFF_BTN_POS,
    WINDOW_HEIGHT,
    WINDOW_NAME,
    WINDOW_WIDTH,
    BotBind,
    GameBind,
    UserBind,
)
from utils import Success
from vision_detector import VisionDetector


def scale_img(img, scale=5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def binarized_to_grayscale(binarized_img):
    # return np.array([[[x, x, x] for x in row] for row in binarized_img], dtype=np.uint8) * 255

    # Ensure the input is a NumPy array for efficient computation
    binarized_img = np.array(binarized_img, dtype=np.uint8)
    # Use broadcasting to replicate the binarized image across the RGB channels
    grayscale_img = np.expand_dims(binarized_img, axis=-1) * 255
    # Stack the single channel image across the three channels (RGB)
    return np.repeat(grayscale_img, 3, axis=-1)


def get_user_pos_from_minimap_text(text):
    pattern = re.compile(r"(\d+),\s*(\d+)")
    match = pattern.search(text)
    if match is None:
        logger.warning(f"Failed to match {text=} {match=}")
        return None
    return tuple(map(int, match.groups()))


class GameController:

    def __init__(
        self,
        vision_detector: VisionDetector,
        start_delay: float = 5,
        saved_credentials_idx: int = 1,
    ):
        self.vision_detector = vision_detector
        self.start_delay = start_delay

        self.saved_credentials_idx = saved_credentials_idx
        self.game_exe_path = r"C:\BOT\ValiumAkademia\valium.exe"  # vm

        self.keyboard = KeyboardController()
        self.keyboard_listener = self._init_keyboard_listener()
        self.keyboard_listener.start()

        self.mouse = MouseController()
        self.mouse_listener = self._init_mouse_listener()
        self.mouse_listener.start()

        self.is_running = True
        self.last_turn = None
        self.attacking = False
        self.mounted = False
        self.is_polymorphed = False

        self.eq_visible = False
        self.minimap_visible = True
        self.map_visible = False
        self.settings_visible = False

        self._start_delay(self.start_delay)
        self._activate_window()

    def press_key(self, key):
        self._check_key_value(key)
        self.keyboard.press(key.value)
        logger.trace(f"Key `{key}` pressed")

    def reset_game_state(self):
        logger.info("Resetting the game state...")
        self.reset_game_visibility_state()
        self.is_running = True
        self.last_turn = None
        self.attacking = False
        self.mounted = False
        self.is_polymorphed = False

    def reset_game_visibility_state(self):
        logger.info("Resetting the game visibility state...")
        self.eq_visible = False
        self.minimap_visible = True
        self.map_visible = False
        self.settings_visible = False

    def _activate_window(self):
        # dummy function to activate the window by clicking on it in safe area
        # window must be visible and not minimized
        safe_point = positions.EXP_CIRCLES_ROI_CENTER
        safe_point_global = self.vision_detector.get_global_pos(safe_point)
        self.click_at(safe_point_global)
        sleep(1)

    def tap_key(self, key, press_time: Optional[float] = None):
        # add extra delay for proper recognition of the key press by game
        self._check_key_value(key.value)
        self.keyboard.press(key.value)
        if press_time is None:
            self._key_release_wait()
        else:
            sleep(press_time)
        self.keyboard.release(key.value)
        self._key_release_wait()
        logger.trace(f"Key `{key}` tapped")

    def release_key(self, key):
        self._check_key_value(key.value)
        self.keyboard.release(key.value)
        logger.trace(f"Key `{key}` released")

    def click(
        self, right: bool = False, times: int = 1, press_time: Optional[float] = None
    ):
        for _ in range(times):
            self._click(right, press_time)

    def _click(self, right: bool = False, press_time: Optional[float] = None):
        button = Button.right if right else Button.left
        self.mouse.press(button)
        if press_time is None:
            self._mouse_btn_release_wait()
        else:
            sleep(press_time)
        self.mouse.release(button)
        logger.trace(f"{'Right' if right else 'Left'} mouse button clicked")

    def catch_fish(self, pos: Tuple[int, int]):
        self.click_at(pos, fast=True)

    def _press_mouse_btn(self, right=False):
        self.mouse.press(Button.right if right else Button.left)

    def _release_mouse_btn(self, right=False):
        self.mouse.release(Button.right if right else Button.left)

    def click_at(self, pos: Tuple[int, int], right=False, times: int = 1, fast=False):
        self.move_cursor_at(pos)
        sleep(0.5)
        self.click(right=right, times=times)

    def get_user_position(
        self, vis_detector: VisionDetector, frame: np.array, indicator_pos
    ) -> Tuple[np.array, Tuple[int, int]]:
        # minimap_user_pos = (WINDOW_WIDTH - 62, 100)
        minimap_user_pos = indicator_pos
        rgb_text_main_color = (255, 215, 76)
        bbox_thickness = 2
        ocr_config = "--psm 7 --oem 1"

        self.move_cursor_at(minimap_user_pos)
        # w, h = 120, 20
        w, h = 60, 10
        x = minimap_user_pos[0] - w
        y = minimap_user_pos[1] - bbox_thickness
        frame = VisionDetector.mark_bbox(frame, x, y, w, h, thickness=bbox_thickness)

        frame_after_mouse_move = vis_detector.capture_frame()
        cropped_user_pos_text_img = VisionDetector.crop_bbox(
            frame_after_mouse_move, x, y, w, h
        )
        logger.warning(f"\n\n{cropped_user_pos_text_img.shape=}")
        cropped_user_pos_text_img = cv2.cvtColor(
            cropped_user_pos_text_img, cv2.COLOR_BGR2RGB
        )

        erode_kernel = np.ones((3, 3), np.uint8)
        upscaled_img = scale_img(cropped_user_pos_text_img, scale=5)
        upscaled_binarized_img = np.where(
            np.all(upscaled_img == rgb_text_main_color, axis=-1), 0, 1
        )
        upscaled_binarized_img_grayscale = binarized_to_grayscale(
            upscaled_binarized_img
        )
        eroded_upscaled_binarized_img = cv2.erode(
            upscaled_binarized_img_grayscale, erode_kernel, iterations=1
        )
        minimap_user_text = pytesseract.image_to_string(
            eroded_upscaled_binarized_img, config=ocr_config
        )
        user_position = get_user_pos_from_minimap_text(minimap_user_text)

        logger.info(f"User position: {user_position}")
        return frame, user_position

    def move_cursor_at(
        self, pos: Tuple[int, int], after_move_wait: Optional[float] = None
    ):
        self.mouse.position = pos
        if after_move_wait is None:
            self._after_mouse_move_wait()
        else:
            sleep(after_move_wait)

    def change_to_channel(self, channel: int, wait_after_change: float = 3):
        logger.info(f"Switching to CH{channel}...")
        ch_ctrl_bind = getattr(Key, f"f{channel}")
        self.press_with(ch_ctrl_bind, Key.ctrl_l)
        sleep(wait_after_change)

    def lure(self):
        logger.info("Luring...")
        self.tap_key(UserBind.PELERYNKA)

    def lure_many(self, uses: int = 2):
        _uses = choices([uses, uses + 1], weights=[0.7, 0.3])[0]
        for _ in range(_uses):
            self.lure()
            sleep(uniform(0.01, 0.02))

    def pickup(self):
        logger.info("Picking up...")
        self.tap_key(GameBind.PICKUP, press_time=uniform(1.3, 1.8))

    def pickup_many(self, uses: int = 2):
        _uses = choices([uses, uses + 1], weights=[0.7, 0.3])[0]
        for _ in range(_uses):
            self.pickup()
            sleep(uniform(0.1, 0.2))

    def pickup_on(self):
        self.press_key(GameBind.PICKUP)

    def pickup_off(self):
        self.release_key(GameBind.PICKUP)

    @staticmethod
    def _release_key_delay():
        return uniform(0.2, 0.25)

    def _key_release_wait(self):
        sleep(self._release_key_delay())

    @staticmethod
    def _release_mouse_btn_delay():
        return uniform(0.1, 0.15)

    def _mouse_btn_release_wait(self):
        sleep(self._release_mouse_btn_delay())

    @staticmethod
    def _after_mouse_move_delay():
        return uniform(0.2, 0.25)

    def _after_mouse_move_wait(self):
        sleep(self._after_mouse_move_delay())

    @staticmethod
    def _start_delay(delay):
        logger.info(f"Starting in {delay} seconds...")
        sleep(delay)

    def toggle_mount(self):
        logger.info("Toggling mount...")
        self.press_with(GameBind.CTRL_MOUNT_KEY, Key.ctrl_l)
        self.mounted = not self.mounted

    def press_with(self, key, with_, press_time: Optional[float] = None):
        with self.keyboard.pressed(with_):
            sleep(0.5)  # wait to register the ctrl press
            self.tap_key(key, press_time=press_time)
        self._key_release_wait()  # to prevent pressing other keys too fast

    def toggle_minimap(self):
        logger.info("Toggling minimap visibility...")
        self.press_with(GameBind.SHIFT_MINIMAP, Key.shift_l)
        self.minimap_visible = not self.minimap_visible

    def hide_minimap(self):
        if self.minimap_visible:
            self.toggle_minimap()

    def show_minimap(self):
        if not self.minimap_visible:
            self.toggle_minimap()

    def turn_randomly(self, turn_press_time: float = 0.6):
        possible_turns = list(
            set([Key.up, Key.down, Key.left, Key.right]) - {self.last_turn}
        )
        turn = choice(possible_turns)
        self.tap_key(turn, press_time=turn_press_time)
        self.last_turn = turn

    def use_boosters(self):
        logger.info("Using boosters...")
        self.tap_key(GameBind.BOOSTERS)

    def toggle_skill(
        self, skill_key: str, reset_animation: bool = True, animation_delay: float = 1
    ):
        if self.mounted:
            logger.warning("Cannot toggle skills while mounted!")
            return
        logger.info(f"Toggling skill `{skill_key}`...")
        self.tap_key(skill_key)
        if reset_animation:
            self._reset_animation()
            return
        sleep(animation_delay)

    def toggle_passive_skills(self, reset_animation: bool = True):
        self.toggle_skill(UserBind.BERSERK, reset_animation=reset_animation)
        self.toggle_skill(UserBind.AURA, reset_animation=reset_animation)

    def start_attack(self):
        if not self.attacking:
            self.press_key(GameBind.ATTACK)
            self.attacking = True

    def stop_attack(self):
        if self.attacking:
            self.release_key(GameBind.ATTACK)
            self.attacking = False

    @staticmethod
    def _check_key_value(key):
        if isinstance(key, str) and key.isupper():
            logger.warning(
                f"Key `{key}` is uppercase! It should be lowercase to be interpreted correctly."
            )

    def _init_keyboard_listener(self):
        keyboard_listener = KeyboardListener(
            on_press=self._on_press, on_release=self._on_release
        )
        return keyboard_listener

    def _init_mouse_listener(self):
        mouse_listener = MouseListener(
            on_click=self._on_click,
        )
        return mouse_listener

    def _on_click(self, x, y, button, pressed):
        frame_x, frame_y = self.vision_detector.get_frame_pos((x, y))
        logger.trace(
            f"Mouse clicked at global: ({x}, {y}) frame: ({frame_x}, {frame_y}) with button `{button}`"
        )

    def _on_press(self, key):
        logger.trace(f"Key `{key}` pressed")
        if key is BotBind.EXIT.value:
            self.exit()

    def _on_release(self, key):
        logger.trace(f"Key `{key}` released")

    def exit(self):
        logger.info("Exiting...")
        self.stop_attack()
        self.keyboard_listener.stop()
        self.mouse_listener.stop()
        self.is_running = False

    def _reset_animation(self):
        self.toggle_mount()
        self.toggle_mount()

    def mount(self):
        if not self.mounted:
            self.summon_horse()
            self.toggle_mount()

    def unmount(self):
        if self.mounted:
            self.toggle_mount()

    def summon_horse(self):
        self.tap_key(UserBind.HORSE)

    def use_polymorph(self):
        self.tap_key(UserBind.MARMUREK, press_time=1)
        sleep(0.5)
        self.is_polymorphed = True

    def polymorph_off(self):
        self.press_with(GameBind.CTRL_POLYMORPH_OFF, Key.ctrl_l, press_time=0.3)
        self.is_polymorphed = False

    def calibrate_camera(self):
        self._reset_camera()

    def calibrate_game_settings(self):
        self.show_settings()

        self.click_at(
            self.vision_detector.get_global_pos(positions.GAME_SYS_SETTINGS_BTN_CENTER)
        )
        self.click_at(
            self.vision_detector.get_global_pos(positions.GSS_CAM_CLOSER_BTN_CENTER)
        )
        self.click_at(
            self.vision_detector.get_global_pos(positions.GSS_FOG_LIGHT_BTN_CENTER)
        )

        self.hide_settings()
        self.show_settings()

        self.click_at(
            self.vision_detector.get_global_pos(positions.GAME_GAME_SETTINGS_BTN_CENTER)
        )
        self.click_at(
            self.vision_detector.get_global_pos(positions.GGS_HIDE_CHAT_BTN_CENTER)
        )
        self.click_at(
            self.vision_detector.get_global_pos(
                positions.GGS_NAME_ITEMS_ONLY_BTN_CENTER
            )
        )

    def toggle_settings(self):
        self.tap_key(GameBind.SETTINGS)
        self.settings_visible = not self.settings_visible

    def show_settings(self):
        if not self.settings_visible:
            self.toggle_settings()

    def hide_settings(self):
        if self.settings_visible:
            self.toggle_settings()

    def _reset_camera(self):
        self.tap_key(GameBind.CAMERA_UP, press_time=1.5)
        self.tap_key(GameBind.CAMERA_ZOOM_OUT, press_time=1.5)

    def move_camera_left(self, press_time: float):
        self.tap_key(GameBind.CAMERA_LEFT, press_time=press_time)

    def move_camera_right(self, press_time: float):
        self.tap_key(GameBind.CAMERA_RIGHT, press_time=press_time)

    def move_camera_down(self, press_time: float):
        self.tap_key(GameBind.CAMERA_DOWN, press_time=press_time)

    def zoomin_camera(self, press_time: float):
        self.tap_key(GameBind.CAMERA_ZOOM_IN, press_time=press_time)

    def zoomout_camera(self, press_time: float):
        self.tap_key(GameBind.CAMERA_ZOOM_OUT, press_time=press_time)

    def steer_randomly(self, press_time: float = 2):
        steer_action = choice(
            [
                self.steer_up_right,
                self.steer_up_left,
                self.steer_down_right,
                self.steer_down_left,
            ]
        )
        steer_action(press_time=press_time)

    def steer_up_right(self, press_time: float = 2):
        self.press_key(GameBind.MOVE_FORWARD)
        self.press_key(GameBind.MOVE_RIGHT)
        sleep(press_time)
        self.release_key(GameBind.MOVE_FORWARD)
        self.release_key(GameBind.MOVE_RIGHT)

    def steer_up_left(self, press_time: float = 2):
        self.press_key(GameBind.MOVE_FORWARD)
        self.press_key(GameBind.MOVE_LEFT)
        sleep(press_time)
        self.release_key(GameBind.MOVE_FORWARD)
        self.release_key(GameBind.MOVE_LEFT)

    def steer_down_right(self, press_time: float = 2):
        self.press_key(GameBind.MOVE_BACKWARD)
        self.press_key(GameBind.MOVE_RIGHT)
        sleep(press_time)
        self.release_key(GameBind.MOVE_BACKWARD)
        self.release_key(GameBind.MOVE_RIGHT)

    def steer_down_left(self, press_time: float = 2):
        self.press_key(GameBind.MOVE_BACKWARD)
        self.press_key(GameBind.MOVE_LEFT)
        sleep(press_time)
        self.release_key(GameBind.MOVE_BACKWARD)
        self.release_key(GameBind.MOVE_LEFT)

    def show_eq(self):
        if not self.eq_visible:
            self.tap_key(GameBind.EQ_MENU)
            self.eq_visible = True

    def hide_eq(self):
        if self.eq_visible:
            self.tap_key(GameBind.EQ_MENU)
            self.eq_visible = False

    def show_eq_slot(self, slot: int):
        if not self.eq_visible:
            self.show_eq()
        if slot not in range(1, 5):
            logger.warning(f"Invalid slot number: {slot}")
            return
        slot_global_center = self.vision_detector.get_global_pos(
            positions.EQ_SLOT_SELECT_BTNS[slot - 1]
        )
        self.click_at(slot_global_center)

    def move_full_butelka_dywizji(self):
        # convention: slot3 - empty bottles, slot4 - full bottles
        self.show_eq_slot(3)
        frame = self.vision_detector.capture_frame()
        bottles_locs = self.vision_detector.detect_butelki_dywizji(frame)
        if len(bottles_locs) == 0:
            logger.warning("No bottles found!. Cannot move the full butelka dywizji!")
            return
        top_left_bottle_loc = bottles_locs[0]
        top_left_bottle_global_loc = self.vision_detector.get_global_pos(
            top_left_bottle_loc
        )
        self.grab_item(top_left_bottle_global_loc)
        self.show_eq_slot(4)
        empty_slots_locs = self.vision_detector.detect_empty_items_slots(frame)
        if len(empty_slots_locs) == 0:
            logger.warning(
                "No empty slots found!. Cannot move the full butelka dywizji!"
            )
            self.click(right=True)  # cancel grabbing
            self.hide_eq()
            return

        empty_slot_loc = empty_slots_locs[0]
        empty_slot_global_loc = self.vision_detector.get_global_pos(empty_slot_loc)
        logger.info(
            f"Moving the full butelka dywizji to the empty slot (location on the screen: {empty_slot_global_loc})..."
        )
        self.put_item(empty_slot_global_loc)
        self.hide_eq()

    def use_next_butelka_dywizji(self):
        # convention: slot3 - empty bottles, slot4 - full bottles
        self.show_eq_slot(3)
        frame = self.vision_detector.capture_frame()
        bottles_locs = self.vision_detector.detect_butelki_dywizji(frame)
        if len(bottles_locs) == 0:
            logger.warning("No empty bottles found!")
            return
        top_left_bottle_loc = bottles_locs[0]
        top_left_bottle_global_loc = self.vision_detector.get_global_pos(
            top_left_bottle_loc
        )
        logger.info(
            f"Using the next butelka dywizji (location on the screen: {top_left_bottle_global_loc})..."
        )
        self.click_at(
            top_left_bottle_global_loc, right=True
        )  # start filling the next bottle
        # confirmation_btn_center = (360, 320)
        confirmation_btn_center_global = self.vision_detector.get_global_pos(
            positions.UZYJ_BUTELKE_CONFIRMATION_BTN
        )
        self.click_at(confirmation_btn_center_global)  # confirm filling the bottle
        self.hide_eq()

    def grab_item(self, item_pos: Tuple[int, int]):
        logger.debug(f"Grabbing the item (location on the screen: {item_pos})...")
        self.click_at(item_pos)
        sleep(0.4)

    def put_item(self, slot_pos: Tuple[int, int]):
        logger.debug(
            f"Putting the item to the slot (location on the screen: {slot_pos})..."
        )
        self.click_at(slot_pos)
        sleep(0.4)

    def login(self):
        # login_btn_center = (400, 300)
        login_btn_global_center = self.vision_detector.get_global_pos(
            positions.LOGIN_BTN_CENTER
        )
        self.load_saved_credentials(idx=self.saved_credentials_idx)
        self.click_at(login_btn_global_center)
        sleep(10)  # wait for character select menu to load
        self.tap_key(Key.enter)  # confirm character selection (first)
        self.reset_game_visibility_state()
        sleep(10)  # wait for the game to load
        logger.info("Logged in successfully!")

    def load_saved_credentials(self, idx: int):
        # first_load_credentials_btn_center: (440, 360)
        logger.info(f"Loading the credentials... ({idx})")
        cred_btn_center = (
            positions.LOAD_CREDENTIAL_BTN_CENTER[0],
            positions.LOAD_CREDENTIAL_BTN_CENTER[1]
            + idx * positions.LOAD_CREDENTIAL_BTN_SPACING,
        )
        cred_btn_global_center = self.vision_detector.get_global_pos(cred_btn_center)
        self.click_at(cred_btn_global_center)

    def open_game(self, load_wait: float = 30):
        logger.info("Opening the game...")
        game_dir = os.path.dirname(
            self.game_exe_path
        )  # Assumes the game's working directory is its location.
        # Set the working directory to the game's directory and run as admin.
        command = rf'Powershell -Command "&{{Set-Location -Path {game_dir}; Start-Process "{self.game_exe_path}" -Verb RunAs}}"'
        os.system(command)
        logger.debug(f"Waiting for the game to load... ({load_wait}s)")
        sleep(load_wait)  # wait for the game to load

    def restart_game(self):
        logger.info("Restarting the game...")
        self.open_game()

    def toggle_map(self):
        logger.debug("Toggling map visibility...")
        self.tap_key(GameBind.MAP, press_time=1)
        self._key_release_wait()  # to prevent pressing other keys too fast
        self.map_visible = not self.map_visible

    def hide_map(self):
        logger.info("Hiding the map...")
        if self.map_visible:
            self.toggle_map()

    def show_map(self):
        logger.info("Showing the map...")
        if not self.map_visible:
            self.toggle_map()

    def teleport_to_polana(self, after_tp_wait: float = 10):
        logger.info("Teleporting to Leśna Polana...")
        self.show_map()
        map_polana_btn_global_center = self.vision_detector.get_global_pos(
            positions.MAP_POLANA_BTN_CENTER
        )
        self.click_at(map_polana_btn_global_center)
        logger.debug(f"Waiting for the teleportation to finish... ({after_tp_wait}s)")
        sleep(after_tp_wait)  # wait for the teleportation to finish
        # update the map visibility state, but there is no need to actually hide the map,
        # because the its hidden after the teleportation
        self.reset_game_visibility_state()

    def idle(
        self,
        time: float,
        capture: bool = False,
        use_boosters: bool = True,
        turn_randomly: bool = False,
        pickup: bool = False,
        lure: bool = False,
        act_seq_wait: Optional[float] = None,
        # ) -> Generator[None | np.ndarray, None, None]:
    ) -> None:
        # yield_frame = None
        t0 = perf_counter()
        logger.info(f"Idling for {time}s...")
        while perf_counter() - t0 <= time:
            seq_t0 = perf_counter()
            if use_boosters:
                self.use_boosters()
            if turn_randomly:
                self.turn_randomly()
            if pickup:
                self.pickup()
            if lure:
                self.lure()
            # if capture:
            #     yield_frame = self.vision_detector.capture_frame()
            # yield yield_frame
            if act_seq_wait is None:
                act_seq_wait = self._idle_act_seq_wait()
            sleep_time = max(act_seq_wait, perf_counter() - seq_t0 + act_seq_wait)
            sleep(sleep_time)
        logger.debug(f"Idling finished")

    def _idle_act_seq_wait(self):
        return uniform(0.05, 0.1)
