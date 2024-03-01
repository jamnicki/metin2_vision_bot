from time import sleep

from game_controller import GameController
from vision_detector import VisionDetector

vision = VisionDetector()
game = GameController(vision_detector=vision, start_delay=3, saved_credentials_idx=1)

for _ in range(20):
    frame = vision.capture_frame()
    if frame is None:
        game.restart_game()
        continue

    if vision.logged_out(frame):
        game.login()
        continue

    sleep(2)
    game.use_boosters()
