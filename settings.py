from enum import Enum
from pathlib import Path, PurePath

from pynput.keyboard import Key

DATA_DIR = Path(PurePath(Path("data").absolute()))
TARGET_TEMPLATES_DIR = DATA_DIR / "target_templates"
EFFECTS = TARGET_TEMPLATES_DIR / "effects"
ITEMS = TARGET_TEMPLATES_DIR / "items"
METINS_DIR = TARGET_TEMPLATES_DIR / "metins"
MODELS_DIR = PurePath("models")
DATASETS_DIR = DATA_DIR / "datasets"


class GameBind(Enum):
    ATTACK = Key.space
    BOOSTERS = Key.f5
    PICKUP = "`"
    CTRL_MOUNT_KEY= "g"
    CTRL_POLYMORPH_OFF = "p"
    CTRL_QUESTS = "q"
    SHIFT_MINIMAP = "m"
    CIECIE_Z_SIODLA = "2"

    MOVE_FORWARD = "w"
    MOVE_BACKWARD = "s"
    MOVE_LEFT = "a"
    MOVE_RIGHT = "d"

    CAMERA_UP = "g"
    CAMERA_DOWN = "t"
    CAMERA_LEFT = "q"
    CAMERA_RIGHT = "e"
    CAMERA_ZOOM_IN = "r"
    CAMERA_ZOOM_OUT = "f"
    
    EQ_MENU = "i"
    MAP = Key.tab
    SETTINGS = Key.esc


class UserBind(Enum):
    HORSE = "1"
    CIECIE_Z_SIODLA = "2"
    PELERYNKA = "3"
    BERSERK = "4"
    AURA = Key.f1
    WIREK = Key.f2
    SZARZA = Key.f3
    MARMUREK = Key.f4


class BotBind(Enum):
    EXIT = Key.end
    VISION_IMSHOW_EXIT = 27  # ESC


class ResourceName(Enum):
    POLYMORPH = "polymorph"
    RUNO_LESNE = "runo_lesne"
    VALIUM_MSG = "valium_msg"
    RUNO_LESNE_DROPPED = "runo_lesne_dropped"
    BUTELKA_DYWIZJI = "butelka_dywizji"
    BUTELKA_DYWIZJI_FILLED_MSG = "butelka_dywizji_filled_msg"
    ZALOGUJ_BUTTON = "zaloguj_button"
    LOADING_ICON = "loading_icon"


WINDOW_NAME = "Akademia Valium.pl"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

GAME_VIEW_POLY_OFF_BTN_POS = (606, 70)  # 800x600

RANDOM_TURN_PROB = 0.2
SPOT_DELAY = 1.1
USE_BOOSTERS_AFTER = 1 * 60
USE_PASSIVE_SKILLS_AFTER = 92

VISION_EFFECTS_BBOX = (5, 33, 268 - 5, 178 - 33)  # 800x600

ICO_POLYMORPH_FPATH = EFFECTS / "polymorph_effect_ico.png"
ICO_RUNO_LESNE_FPATH = ITEMS / "runo_lesne_ico.png"
RUNO_LESNE_DROPPED_FPATH = ITEMS / "runo_lesne_dropped.png"
TEMPLATE_BUTELKA_DYWIZJI_FPATH = ITEMS / "butelka_dywizji.png"
TEMPLATE_VALIUM_MSG_FPATH = TARGET_TEMPLATES_DIR / "valium_message_lowbar.png"
BUTELKA_DYWIZJI_FILLED_MSG_FPATH = TARGET_TEMPLATES_DIR / "butelka_dywizji_filled_msg.png"
ZALOGUJ_BUTTON_FPATH = TARGET_TEMPLATES_DIR / "zaloguj_button.png"
LOADING_ICON_FPATH = TARGET_TEMPLATES_DIR / "valium_akademia_loading.png"

CAP_MAX_FPS = 30
WINDOW_NOT_FOUND_EXIT_DELAY = 4
