from pathlib import PurePath
from time import sleep
from typing import Tuple

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR/tesseract.exe'

import cv2
import numpy as np
import win32con
import win32gui
import win32ui
from loguru import logger

from settings import (
    BUTELKA_DYWIZJI_FILLED_MSG_FPATH,
    ICO_POLYMORPH_FPATH,
    ICO_RUNO_LESNE_FPATH,
    RUNO_LESNE_DROPPED_FPATH,
    TEMPLATE_BUTELKA_DYWIZJI_FPATH,
    TEMPLATE_VALIUM_MSG_FPATH,
    VISION_EFFECTS_BBOX,
    WINDOW_HEIGHT,
    WINDOW_NAME,
    WINDOW_WIDTH,
    ZALOGUJ_BUTTON_FPATH,
    BotBind,
    ResourceName,
)
import positions


class WindowNotFoundError(ValueError):
    pass


class VisionDetector:

    def __init__(self, preview: bool = True):
        self.preview = preview
        self.target_templates = {
            ResourceName.POLYMORPH.value: self._imread_template(ICO_POLYMORPH_FPATH),
            ResourceName.RUNO_LESNE.value: self._imread_template(ICO_RUNO_LESNE_FPATH),
            ResourceName.RUNO_LESNE_DROPPED.value: self._imread_binarize_template(RUNO_LESNE_DROPPED_FPATH, min_grayscale_threshold=254),  # 254 to capture only white pixels - item name
            ResourceName.VALIUM_MSG.value: self._imread_template(TEMPLATE_VALIUM_MSG_FPATH),
            ResourceName.BUTELKA_DYWIZJI_FILLED_MSG.value: self._imread_template(BUTELKA_DYWIZJI_FILLED_MSG_FPATH),
            ResourceName.BUTELKA_DYWIZJI.value: self._imread_gray_template(TEMPLATE_BUTELKA_DYWIZJI_FPATH),
            ResourceName.ZALOGUJ_BUTTON.value: self._imread_template(ZALOGUJ_BUTTON_FPATH),
        }

        self.hwnd = self.get_window_handler()
        
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        logger.info(f"Window size: {self.w=} {self.h=}")

        self.window_x, self.window_y = window_rect[:2]

        # account for the window border and titlebar and cut them off
        self.border_pixels = 8
        self.titlebar_pixels = 30
        self.w = self.w - (self.border_pixels * 2)
        self.h = self.h - self.titlebar_pixels - self.border_pixels
        self.cropped_x = self.border_pixels
        self.cropped_y = self.titlebar_pixels
        logger.info(f"Window size after crop: {self.w=} {self.h=}")

        self.center = (self.w // 2, self.h // 2)

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = self.window_x + self.cropped_x
        self.offset_y = self.window_y + self.cropped_y

    def get_window_handler(self):
        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        if hwnd == 0:
            raise WindowNotFoundError(f"Window '{WINDOW_NAME}' not found. Is the game running?")
        return hwnd

    def reload_window_handler(self):
        self.hwnd = self.get_window_handler()

    def get_global_pos(self, in_frame_pos: Tuple[int, int]):
        return (in_frame_pos[0] + self.offset_x, in_frame_pos[1] + self.offset_y)

    def get_frame_pos(self, global_pos: Tuple[int, int]):
        return (global_pos[0] - self.offset_x, global_pos[1] - self.offset_y)

    def is_polymorphed(self, frame: np.ndarray) -> Tuple[np.array, bool]:
        last_effects_ROI = self.crop_effects_ROI(frame)
        polymorphed, confidence, loc = self._find_by_template(last_effects_ROI, ResourceName.POLYMORPH, confidence_threshold=0.95)
        loc = (loc[0] + VISION_EFFECTS_BBOX[0], loc[1] + VISION_EFFECTS_BBOX[1])  # translate from ROI to global view coords
        logger.debug(f"Polymorph {'ON' if polymorphed else 'OFF'}\t{confidence=:.2f} {loc=}")
        if polymorphed and self.preview:
            frame = self.mark_polymorph_detection(frame, confidence, loc)
        return frame, polymorphed

    def _find_by_template(self, frame: np.ndarray, template: ResourceName, confidence_threshold: float) -> Tuple[bool, float, Tuple[int, int]]:
        result = cv2.matchTemplate(frame, self.target_templates[template.value], cv2.TM_CCOEFF_NORMED)
        _, confidence, _, loc = cv2.minMaxLoc(result)
        active = confidence >= confidence_threshold
        logger.debug(f"{template.value} {'DETECTED' if active else 'NOT DETECTED'}\t{confidence=:.2f} {loc=}")
        return active, confidence, loc

    def _find_many_by_template(self, frame: np.ndarray, template: ResourceName, confidence_threshold: float) -> Tuple[Tuple[int, int]]:
        # returns a list of tuples with (x, y) coordinates of the detected objects
        result = cv2.matchTemplate(frame, self.target_templates[template.value], cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= confidence_threshold)
        return tuple(zip(*locs[::-1]))

    def detect_runo_lesne(self, frame: np.ndarray) -> Tuple[bool, float, Tuple[int, int]]:
        detected, confidence, loc = self._find_by_template(frame, ResourceName.RUNO_LESNE, confidence_threshold=0.8)
        item_loc_center = self.get_bbox_center(*loc, *self.get_img_wh(self.target_templates[ResourceName.RUNO_LESNE.value]))
        logger.debug(f"Runo lesne {'DETECTED' if detected else 'NOT DETECTED'}\t{confidence=:.2f} {loc=}")
        return detected, confidence, item_loc_center

    def detect_runo_lesne_dropped(self, frame: np.ndarray) -> Tuple[bool, float, Tuple[int, int]]:
        binarized_frame = self.binarize_img(frame, min_grayscale_threshold=254)  # 254 to capture only white pixels - item name
        detected, confidence, loc = self._find_by_template(frame, ResourceName.RUNO_LESNE_DROPPED, confidence_threshold=0.8)
        item_loc_center = self.get_bbox_center(*loc, *self.get_img_wh(self.target_templates[ResourceName.RUNO_LESNE_DROPPED.value]))
        logger.debug(f"Runo lesne dropped {'DETECTED' if detected else 'NOT DETECTED'}\t{confidence=:.2f} {loc=}")
        return detected, confidence, item_loc_center

    def detect_butelki_dywizji(self, frame: np.ndarray) -> None:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_wh = self.get_img_wh(self.target_templates[ResourceName.BUTELKA_DYWIZJI.value])
        locs = self._find_many_by_template(frame_gray, ResourceName.BUTELKA_DYWIZJI, confidence_threshold=0.9)
        locs_centers = [self.get_bbox_center(*loc, *template_wh) for loc in locs]
        return locs_centers

    def detect_empty_items_slots(self, frame: np.ndarray) -> None:
        # tested ONLY for butelki dywizji, może nie działać dla innych itemów

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold the image to get the slots as white on a black background.
        _, binary_image = cv2.threshold(frame_gray, 15, 255, cv2.THRESH_BINARY_INV)
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out contours that don't match the size and shape of an inventory slot
        slot_contours = []
        for contour in contours:
            # Calculate contour area and filter out small areas
            area = cv2.contourArea(contour)
            # if area < 400 or area > 1000:  # Inventory slots should fall within this range
            #     continue
            if area < 400 or area > 1800:  # Inventory slots should fall within this range
                continue
            # Get the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Inventory slots are more or less square, with the width and height being similar
            if 0.8 < w/h < 1.2:
                # Check if the slot is in the lower right part (which is the inventory area)
                # if (630 < x) and (210 < y < 550):  # 800x600
                if (630 < x) and (140 < y < 480):  # 800x530
                    slot_contours.append(contour)
        # Calculate centroids for the filtered contours
        centroids = []
        for contour in slot_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
        return centroids

    def get_img_wh(self, img: np.ndarray) -> Tuple[int, int]:
        return img.shape[:2][::-1]

    def frame_contains_valium_message(self, frame: np.ndarray) -> bool:
        # to prevent message from being captured instead of dungeon message
        detected, confidence, loc = self._find_by_template(frame, ResourceName.VALIUM_MSG, confidence_threshold=0.8)
        logger.debug(f"Valium message {'DETECTED' if detected else 'NOT DETECTED'}\t{confidence=:.2f} {loc=}")
        return detected

    def detect_butelka_dywizji_filled_message(self, frame: np.ndarray) -> bool:
        detected, confidence, loc = self._find_by_template(frame, ResourceName.BUTELKA_DYWIZJI_FILLED_MSG, confidence_threshold=0.8)
        if detected:
            logger.debug(f"Butelka dywizji has been filled!\t{confidence=:.2f} {loc=}")
        return detected

    def detect_login_button(self, frame: np.ndarray) -> Tuple[bool, float, Tuple[int, int]]:
        detected, confidence, loc = self._find_by_template(frame, ResourceName.ZALOGUJ_BUTTON, confidence_threshold=0.8)
        logger.debug(f"Logging menu {'DETECTED' if detected else 'NOT DETECTED'}\t{confidence=:.2f} {loc=}")
        btn_center = self.get_bbox_center(*loc, *self.get_img_wh(self.target_templates[ResourceName.ZALOGUJ_BUTTON.value]))
        return detected, confidence, btn_center

    def logged_out(self, frame: np.ndarray) -> bool:
        return self.detect_login_button(frame)[0]

    def capture_frame(self) -> np.ndarray | None:
        try:
            self.reload_window_handler()
        except WindowNotFoundError as e:
            logger.error(e)
            return None

        # Move the window to the specified position (initial position from first run)
        # SetWindowPos parameters: HWND, HWND insert after, x, y, cx, cy, flags
        # cx and cy are the width and height, set to 0 to ignore
        # SWP_NOSIZE: Retains the current size (ignores the cx and cy parameters).
        # SWP_NOZORDER: Retains the current Z order (ignores the HWND insert after parameter).
        win32gui.SetWindowPos(self.hwnd, None, self.window_x, self.window_y, 0, 0, win32con.SWP_NOZORDER | win32con.SWP_NOSIZE)

        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        raw_frame = np.fromstring(signedIntsArray, dtype='uint8')
        raw_frame.shape = (self.h, self.w, 4)

        # free caputured resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        raw_frame = self.drop_alpha_channel(raw_frame)
        raw_frame = np.ascontiguousarray(raw_frame)  # make image C_CONTIGUOUS to avoid errors in cv2
        return raw_frame

    def mark_polymorph_detection(self, frame: np.ndarray, confidence: float, loc: Tuple[int, int]):
        return self.mark_effect_detection(frame, ResourceName.POLYMORPH, confidence, loc, color=(247, 22, 135))

    @staticmethod
    def scale_frame(frame: np.ndarray, scale: float = 0.7):
        # ex. 0.6 -> 60% of the original size (downscale)
        # ex. 1.4 -> 140% of the original size (upscale)   
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        div = (width, height)
        return cv2.resize(frame, div, interpolation=cv2.INTER_AREA)

    @staticmethod
    def drop_alpha_channel(frame: np.ndarray) -> np.array:
        return frame[...,:3]

    @staticmethod
    def crop_bbox(frame: np.ndarray, x: int, y: int, w: int, h: int) -> cv2.UMat:
        return frame[y:y+h, x:x+w]

    @staticmethod
    def get_bbox_center(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return (x + w // 2, y + h // 2)

    @staticmethod
    def crop_effects_ROI(frame: np.ndarray) -> cv2.UMat:
        """Crop extras region (boosters, passive skills, etc.)"""
        return VisionDetector.crop_bbox(frame, *VISION_EFFECTS_BBOX)

    @staticmethod
    def mark_bbox(frame: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int] = (0, 255, 0),
                  thickness=2
    ) -> cv2.UMat:
        return cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

    def mark_effects_ROI(self, frame: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness=1
    ) -> cv2.UMat:
        x, y, w, h = VISION_EFFECTS_BBOX
        VisionDetector.mark_bbox(frame, x, y, w, h, color, thickness)
    
    def mark_effect_detection(self, frame: np.ndarray, effect: ResourceName, confidence: float, loc: Tuple[int, int],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness=2
    ) -> cv2.UMat:
        x, y = loc
        w, h = self.target_templates[effect.value].shape[:2]
        bbox_label = f"{effect.value} ({confidence:.2f})"
        frame = VisionDetector.draw_bbox_label(frame, bbox_label, (x, y), color)
        frame = VisionDetector.mark_bbox(frame, x, y, w, h, color, thickness)
        return frame

    def _imread_template(self, png_fpath: PurePath) -> cv2.UMat:
        img = cv2.imread(str(png_fpath), cv2.IMREAD_UNCHANGED)
        return self.drop_alpha_channel(img)

    def _imread_gray_template(self, png_fpath: PurePath) -> cv2.UMat:
        img = self._imread_template(png_fpath)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _imread_binarize_template(self, png_fpath: PurePath, min_grayscale_threshold: int) -> cv2.UMat:
        img = self._imread_template(png_fpath)
        binarized_template_img = self.binarize_img(img, min_grayscale_threshold)
        return binarized_template_img

    def binarize_img(self, img: np.ndarray, min_grayscale_threshold: int) -> np.array:
        _, binarized_img = cv2.threshold(img, min_grayscale_threshold, 255, cv2.THRESH_BINARY)
        return binarized_img

    @staticmethod
    def draw_bbox_label(frame: np.ndarray, label: str, bbox_pos: Tuple[int, int], bbox_color: Tuple[int, int, int],
                        text_color: Tuple[int, int, int] = (255, 255, 255),
                        text_scale: float = 0.4
    ) -> cv2.UMat:
        bbox_x, bbox_y = bbox_pos
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        frame = cv2.rectangle(frame, (bbox_x, bbox_y - 20), (bbox_x + text_w, bbox_y), bbox_color, -1)
        frame = cv2.putText(frame, label, (bbox_x, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1)
        return frame

    def exit(self):
        cv2.destroyAllWindows()


    def show_preview(self, frame: np.ndarray):
        def _log_imshow_mouse_pos(event, x, y, flags, param):
            preview_mouse_pos = (x, y)
            global_mouse_pos = self.get_global_pos(preview_mouse_pos)
            logger.debug(f"Preview mouse position: {preview_mouse_pos}")
            logger.debug(f"Global mouse position: {global_mouse_pos}")

        imshow_win_name = (
            f"Capturing Preview - Press [{BotBind.EXIT.value}] or"
            f" [{self.get_cv2_keyname(BotBind.VISION_IMSHOW_EXIT.value)}] to exit"
        )
        if frame is None:
            logger.warning("No frame to show. Skipping...")
            return

        cv2.imshow(imshow_win_name, frame)
        cv2.setMouseCallback(imshow_win_name, _log_imshow_mouse_pos)
        key = cv2.waitKey(1_000) % 0xFF
        if key == BotBind.VISION_IMSHOW_EXIT.value or key == BotBind.EXIT.value:
            logger.info("Vision preview exited by user.")
            vision.exit()

    @staticmethod
    def get_cv2_keyname(code: int) -> str:
        key_map = {27: "esc"}
        if code in key_map:
            return key_map[code]
        # For printable ASCII characters, return the character itself
        if 0 <= code <= 127:
            return chr(code)
        # If the key code is not recognized, return a default message
        return "<unkown-key>"

    @staticmethod
    def get_dungeon_message(frame: np.ndarray) -> str:
        # msg_bbox = (130, 108, 540, 16)
        msg_text_color = (242, 231, 193)

        min_grayscale_threshold = 200
        upscale = 10
        oem = 1
        psm = 7  # implies that we are treating the ROI as a single line of text
        ocr_config = f'-l pol --psm {psm} --oem {oem}'
        kernel = np.ones((3,3), np.uint8)

        cropped_msg_img = VisionDetector.crop_bbox(frame, *positions.DUNGEON_MSG_BBOX)
        cropped_msg_img_grayscale = cv2.cvtColor(cropped_msg_img, cv2.COLOR_BGR2GRAY)
        upscaled_img = VisionDetector.scale_frame(cropped_msg_img_grayscale, scale=upscale)
        _, upscaled_binarized_img = cv2.threshold(upscaled_img, min_grayscale_threshold, 255, cv2.THRESH_BINARY)
        upscaled_binarized_img = cv2.bitwise_not(upscaled_binarized_img)  # swap 0 - 255 to match the text color
        eroded_upscaled_binarized_img = cv2.erode(upscaled_binarized_img, kernel, iterations = 1)

        text = pytesseract.image_to_string(eroded_upscaled_binarized_img, config=ocr_config)
        logger.debug(f"Dungeon message: {text=}")
        return text.strip()

    def get_target_text(self, frame: np.ndarray) -> str:
        # target_name_bbox = (255, 20, 150, 20)
        roi = VisionDetector.crop_bbox(frame, *positions.TARGET_NAME_BBOX)

        cv2.imwrite("target_menu_roi.png", roi)

        min_grayscale_threshold = 80
        upscale = 10
        oem = 1
        psm = 7  # implies that we are treating the ROI as a single line of text
        ocr_config = f'-l pol --psm {psm} --oem {oem}'

        cropped_roi_grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        upscaled_img = VisionDetector.scale_frame(cropped_roi_grayscale, scale=upscale)

        _, upscaled_binarized_img = cv2.threshold(upscaled_img, min_grayscale_threshold, 255, cv2.THRESH_BINARY)
        upscaled_binarized_img = cv2.bitwise_not(upscaled_binarized_img)  # swap 0 - 255 to match the text color

        text = pytesseract.image_to_string(upscaled_binarized_img, config=ocr_config)
        logger.debug(f"Target name: {text=}")
        return text

    @staticmethod
    def fill_non_clickable_wth_black(frame: np.ndarray) -> np.array:
        fill_color = (0, 0, 0)
        thickness = -1

        # # 800 x 600
        # frame = cv2.rectangle(frame, (0, 0), (260, 90), fill_color, thickness)  # mask effects roi
        # frame = cv2.rectangle(frame, (0, 170), (120, 350), fill_color, thickness)  # mask quests roi
        # frame = cv2.rectangle(frame, (0, 600), (800, 530), fill_color, thickness)  # mask low bar
        # frame = cv2.rectangle(frame, (640, 600), (800, 0), fill_color, thickness)  # mask minimap; right bar
        # frame = cv2.rectangle(frame, (100, 530), (700, 500), fill_color, thickness)  # mask rest of the chat

        # 800 x 530 (y * 0.83 to local)
        frame = cv2.rectangle(frame, (0, 0), (260, 80), fill_color, thickness)  # mask effects roi
        frame = cv2.rectangle(frame, (0, 150), (120, 309), fill_color, thickness)  # mask quests roi
        frame = cv2.rectangle(frame, (0, 530), (800, 460), fill_color, thickness)  # mask low bar
        frame = cv2.rectangle(frame, (640, 530), (800, 0), fill_color, thickness)  # mask minimap; right bar
        frame = cv2.rectangle(frame, (100, 468), (700, 420), fill_color, thickness)  # mask rest of the chat

        return frame
