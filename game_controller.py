import os
import re
from random import choices, uniform
from time import perf_counter, sleep
from typing import Optional, Tuple

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR/tesseract.exe'

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

from settings import (
    GAME_VIEW_POLY_OFF_BTN_POS,
    WINDOW_HEIGHT,
    WINDOW_NAME,
    WINDOW_WIDTH,
    BotBind,
    GameBind,
    UserBind,
)
from vision_detector import VisionDetector
import positions


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

def get_user_pos_from_minimap_text(text):
    pattern = re.compile(r"(\d+),\s*(\d+)")
    match = pattern.search(text)
    if match is None:
        logger.warning(f"Failed to match {text=} {match=}")
        return None
    return tuple(map(int, match.groups()))


class GameController:

    def __init__(self, vision_detector: VisionDetector, start_delay: float = 5, saved_credentials_idx: int = 1):
        self.vision_detector = vision_detector
        self.start_delay = start_delay

        self._start_delay(self.start_delay)
        
        self.saved_credentials_idx = saved_credentials_idx
        # self.game_exe_path = r"D:\Gry\ValiumAkademia\valium.exe"  # local
        self.game_exe_path = r"C:\ValiumAkademia\valium.exe"  # vbox

        self.keyboard = KeyboardController()
        self.keyboard_listener = self._init_keyboard_listener()
        self.keyboard_listener.start()

        self.mouse = MouseController()
        self.mouse_listener = self._init_mouse_listener()
        self.mouse_listener.start()

        self.is_running = True
        self.last_turn = None
        self.attacking = False
        self.mounted = False  # TODO: implement mounted detection
        self.items_in_range = True  # TODO: implement items detection
        self.is_polymorphed = False

        self.eq_visible = False
        self.minimap_visible = True
        self.map_visible = False

    def press_key(self, key):
        self._check_key_value(key)
        self.keyboard.press(key.value)
        logger.trace(f"Key `{key}` pressed")

    def _reset_controller_attributes(self):
        self.is_running = True
        self.last_turn = None
        self.attacking = False
        self.mounted = False
        self.items_in_range = True
        self.is_polymorphed = False
        self.eq_visible = False
        self.minimap_visible = False

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
    
    def click(self, right: bool = False, times: int = 1, press_time: Optional[float] = None):
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

    def _press_mouse_btn(self, right=False):
        self.mouse.press(Button.right if right else Button.left)

    def _release_mouse_btn(self, right=False):
        self.mouse.release(Button.right if right else Button.left)

    def click_at(self, pos: Tuple[int, int], right=False, times: int = 1):
        self.move_cursor_at(pos)
        sleep(0.3)
        self.click(right=right, times=times)

    def get_user_position(self, vis_detector: VisionDetector, frame: np.array, indicator_pos) -> Tuple[np.array, Tuple[int, int]]:
        # minimap_user_pos = (WINDOW_WIDTH - 62, 100)
        minimap_user_pos = indicator_pos
        rgb_text_main_color = (255, 215, 76)
        bbox_thickness = 2
        ocr_config = '--psm 7 --oem 1'

        self.move_cursor_at(minimap_user_pos)
        # w, h = 120, 20
        w, h = 60, 10
        x = minimap_user_pos[0] - w
        y = minimap_user_pos[1] - bbox_thickness
        frame = VisionDetector.mark_bbox(frame, x, y, w, h, thickness=bbox_thickness)

        frame_after_mouse_move = vis_detector.capture_frame()
        cropped_user_pos_text_img = VisionDetector.crop_bbox(frame_after_mouse_move, x, y, w, h)
        logger.warning(f"\n\n{cropped_user_pos_text_img.shape=}")
        cropped_user_pos_text_img = cv2.cvtColor(cropped_user_pos_text_img, cv2.COLOR_BGR2RGB)

        erode_kernel = np.ones((3,3), np.uint8)
        upscaled_img = scale_img(cropped_user_pos_text_img, scale=5)
        upscaled_binarized_img = np.where(np.all(upscaled_img == rgb_text_main_color, axis=-1), 0, 1)
        upscaled_binarized_img_grayscale = binarized_to_grayscale(upscaled_binarized_img)
        eroded_upscaled_binarized_img = cv2.erode(upscaled_binarized_img_grayscale, erode_kernel, iterations = 1)
        minimap_user_text = pytesseract.image_to_string(eroded_upscaled_binarized_img, config=ocr_config)
        user_position = get_user_pos_from_minimap_text(minimap_user_text)

        logger.info(f"User position: {user_position}")
        return frame, user_position

    def move_cursor_at(self, pos: Tuple[int, int], after_move_wait: Optional[float] = None):
        self.mouse.position = pos
        if after_move_wait is None:
            self._after_mouse_move_wait()
        else:
            sleep(after_move_wait)

    def change_to_channel(self, channel: int, wait_after_change: float = 2):
        logger.info(f"Switching to CH{channel}...")
        with self.keyboard.pressed(Key.ctrl_l):
            self.tap_key(getattr(Key, f"f{channel}"))
        self._key_release_wait()  # to prevent pressing other keys too fast
        sleep(wait_after_change)

    def lure(self):
        logger.info("Luring...")
        self.tap_key(UserBind.PELERYNKA)

    def lure_many(self, uses: int = 2):
        _uses = choices([uses, uses + 1, uses + 2], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(_uses):
            self.lure()
            sleep(uniform(0.01, 0.02))

    def pickup(self):
        if self.items_in_range:
            logger.info("Picking up...")
            self.tap_key(GameBind.PICKUP, press_time=uniform(1.3, 1.8))

    def pickup_many(self, uses: int = 1):
        _uses = choices([uses, uses + 1, uses + 2], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(uses):
            self.pickup()
            sleep(uniform(0.01, 0.02))

    def pickup_on(self):
        self.press_key(GameBind.PICKUP)

    def pickup_off(self):
        self.release_key(GameBind.PICKUP)

    @staticmethod
    def _release_key_delay():
        return uniform(0.15, 0.2)

    def _key_release_wait(self):
        sleep(self._release_key_delay())

    @staticmethod
    def _release_mouse_btn_delay():
        return uniform(0.15, 0.2)

    def _mouse_btn_release_wait(self):
        sleep(self._release_mouse_btn_delay())
    
    @staticmethod
    def _after_mouse_move_delay():
        return uniform(0.15, 0.2)

    def _after_mouse_move_wait(self):
        sleep(self._after_mouse_move_delay())
    

    @staticmethod
    def _start_delay(delay):
        logger.info(f"Starting in {delay} seconds...")
        sleep(delay)

    def toggle_mount(self):
        logger.info("Toggling mount...")
        with self.keyboard.pressed(Key.ctrl_l):
            self.tap_key(GameBind.CTRL_MOUNT_KEY)
        self._key_release_wait()  # to prevent pressing other keys too fast

    def toggle_minimap(self):
        logger.info("Toggling minimap visibility...")
        with self.keyboard.pressed(Key.shift_l):
            self.tap_key(GameBind.SHIFT_MINIMAP)
        self._key_release_wait()  # to prevent pressing other keys too fast
        self.minimap_visible = not self.minimap_visible
    
    def hide_minimap(self):
        if self.minimap_visible:
            self.toggle_minimap()

    def show_minimap(self):
        if not self.minimap_visible:
            self.toggle_minimap()

    def turn_randomly(self, turn_press_time: float = 0.4, weights: Optional[Tuple[float, float, float, float]] = None):
        if weights is None:
            weights = [0.4, 0.4, 0.1, 0.1]
        possible_turns = [Key.up, Key.down, Key.left, Key.right]
        turn = choices(possible_turns, weights=weights)[0]
        if self.last_turn is None:
            self.tap_key(turn, press_time=turn_press_time)

        logger.info("Turning randomly...")
        turn = choices(possible_turns, weights=weights)[0]
        while turn == -1:
            turn = choices(possible_turns, weights=weights)[0]
        self.tap_key(turn, press_time=turn_press_time)
        self.last_turn = turn

    def use_boosters(self):
        logger.info("Using boosters...")
        self.tap_key(GameBind.BOOSTERS)

    def toggle_skill(self, skill_key: str, reset_animation: bool = True, animation_delay: float = 1):
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
            logger.warning(f"Key `{key}` is uppercase! It should be lowercase to be interpreted correctly.")

    def _init_keyboard_listener(self):
        keyboard_listener = KeyboardListener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        return keyboard_listener

    def _init_mouse_listener(self):
        mouse_listener = MouseListener(
            on_click=self._on_click,
        )
        return mouse_listener

    def _on_click(self, x, y, button, pressed):
        frame_x, frame_y = self.vision_detector.get_frame_pos((x, y))
        logger.trace(f"Mouse clicked at global: ({x}, {y}) frame: ({frame_x}, {frame_y}) with button `{button}`")

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
            self.toggle_mount()
            self.mounted = True

    def unmount(self):
        if self.mounted:
            self.toggle_mount()
            self.mounted = False

    def use_polymorph(self):
        self.tap_key(UserBind.MARMUREK, press_time=1)
        sleep(0.5)

    def polymorph_off(self):
        # if self.is_polymorphed:
        #     with self.keyboard.pressed(Key.ctrl_l):
        #         self.tap_key(GameBind.CTRL_POLYMORPH_OFF, press_time=0.3)
        #         self._key_release_wait()
        #     self._key_release_wait()
        #     self.is_polymorphed = False
        global_polymorph_off_btn_pos = self.vision_detector.get_global_pos(GAME_VIEW_POLY_OFF_BTN_POS)
        self.show_eq()
        sleep(0.1)
        self.click_at(global_polymorph_off_btn_pos)  # idk ctrl+p nie dziaua
        sleep(0.1)
        self.hide_eq()

    def calibrate_camera(self):
        self._reset_camera()

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
        slot_global_center = self.vision_detector.get_global_pos(positions.EQ_SLOT_SELECT_BTNS[slot - 1])
        self.click_at(slot_global_center)

    def move_full_butelka_dywizji(self):
        # convention: slot3 - empty bottles, slot4 - full bottles
        self.show_eq_slot(3)
        frame = self.vision_detector.capture_frame()
        bottles_locs = self.vision_detector.detect_butelki_dywizji(frame)
        top_left_bottle_loc = bottles_locs[0]
        top_left_bottle_global_loc = self.vision_detector.get_global_pos(top_left_bottle_loc)
        self.grab_item(top_left_bottle_global_loc)
        self.show_eq_slot(4)
        empty_slots_locs = self.vision_detector.detect_empty_items_slots(frame)
        if len(empty_slots_locs) == 0:
            self.click(right=True)  # cancel grabbing
            self.hide_eq()
            return

        empty_slot_loc = empty_slots_locs[0]  # 
        empty_slot_global_loc = self.vision_detector.get_global_pos(empty_slot_loc)
        self.put_item(empty_slot_global_loc)
        self.hide_eq()

    def use_next_butelka_dywizji(self):
        # convention: slot3 - empty bottles, slot4 - full bottles
        self.show_eq_slot(3)
        frame = self.vision_detector.capture_frame()
        bottles_locs = self.vision_detector.detect_butelki_dywizji(frame)
        top_left_bottle_loc = bottles_locs[0]
        top_left_bottle_global_loc = self.vision_detector.get_global_pos(top_left_bottle_loc)
        self.click_at(top_left_bottle_global_loc, right=True)  # start filling the next bottle
        # confirmation_btn_center = (360, 320)
        confirmation_btn_center_global = self.vision_detector.get_global_pos(positions.UZYJ_BUTELKE_CONFIRMATION_BTN)
        self.click_at(confirmation_btn_center_global)  # confirm filling the bottle
        self.hide_eq()

    def grab_item(self, item_pos: Tuple[int, int]):
        self.click_at(item_pos)
        sleep(0.3)

    def put_item(self, slot_pos: Tuple[int, int]):
        self.click_at(slot_pos)
        sleep(0.3)

    def login(self):
        # login_btn_center = (400, 300)
        login_btn_global_center = self.vision_detector.get_global_pos(positions.LOGIN_BTN_CENTER)
        self.load_saved_credentials(idx=self.saved_credentials_idx)
        self.click_at(login_btn_global_center)
        sleep(5)  # wait for character select menu to load
        self.tap_key(Key.enter)  # confirm character selection (first)
        sleep(5)  # wait for the game to load
        logger.info("Logged in successfully!")

    def load_saved_credentials(self, idx: int):
        # first_load_credentials_btn_center: (440, 360)
        cred_btn_center = (
            positions.LOAD_CREDENTIAL_BTN_CENTER[0],
            positions.LOAD_CREDENTIAL_BTN_CENTER[1] + idx * positions.LOAD_CREDENTIAL_BTN_SPACING
        )
        cred_btn_global_center = self.vision_detector.get_global_pos(cred_btn_center)
        self.click_at(cred_btn_global_center)
        logger.info(f"Credentials loaded successfully\t{idx}")

    def open_game(self):
        game_dir = os.path.dirname(self.game_exe_path)  # Assumes the game's working directory is its location.
        # Set the working directory to the game's directory and run as admin.
        command = fr'Powershell -Command "&{{Set-Location -Path {game_dir}; Start-Process "{self.game_exe_path}" -Verb RunAs}}"'
        os.system(command)
        sleep(30)  # wait for the game to load

    def restart_game(self):
        self.open_game()
        # self._reset_controller_attributes()

    def toggle_map(self):
        logger.info("Toggling map visibility...")
        self.tap_key(GameBind.MAP, press_time=1)
        self._key_release_wait()  # to prevent pressing other keys too fast
        self.map_visible = not self.map_visible
    
    def hide_map(self):
        if self.map_visible:
            self.toggle_map()

    def show_map(self):
        if not self.map_visible:
            self.toggle_map()

    def teleport_to_polana(self):
        self.show_map()
        map_polana_btn_global_center = self.vision_detector.get_global_pos(positions.MAP_POLANA_BTN_CENTER)
        self.click_at(map_polana_btn_global_center)
        sleep(6)  # wait for the teleportation to finish
        # update the map visibility state, but there is no need to actually hide the map,
        # because the its hidden after the teleportation
        self.map_visible = False
