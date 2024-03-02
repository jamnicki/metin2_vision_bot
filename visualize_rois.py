import cv2
from vision_detector import VisionDetector
import positions
from matplotlib import pyplot as plt
from settings import GAME_VIEW_POLY_OFF_BTN_POS


vision = VisionDetector()
frame = vision.capture_frame()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


out = frame_rgb

out = VisionDetector.fill_non_clickable_wth_black(frame_rgb)
out = VisionDetector.mark_bbox(out, *positions.DUNGEON_MSG_BBOX)
out = VisionDetector.mark_bbox(out, *positions.TARGET_NAME_BBOX)

plt.figure(figsize=(16, 9))
plt.imshow(out)

# mapka lesna polana
plt.plot([positions.MAP_POLANA_BTN_CENTER[0]], [positions.MAP_POLANA_BTN_CENTER[1]], marker="x", markersize=20, linewidth=1, color="red")

# # butelki, empty slots
slots_centroids = vision.detect_empty_items_slots(frame_rgb)
for centroid in slots_centroids:
    plt.plot([centroid[0]], [centroid[1]], marker="o", markersize=10, linewidth=1, color="green")

# # buttons eq slots and confirmation menu
plt.plot([positions.EQ_SLOT1_CENTER[0]], [positions.EQ_SLOT1_CENTER[1]], marker="x", markersize=10, linewidth=1, color="lightblue")
plt.plot([positions.EQ_SLOT2_CENTER[0]], [positions.EQ_SLOT2_CENTER[1]], marker="x", markersize=10, linewidth=1, color="red")
plt.plot([positions.EQ_SLOT3_CENTER[0]], [positions.EQ_SLOT3_CENTER[1]], marker="x", markersize=10, linewidth=1, color="pink")
plt.plot([positions.EQ_SLOT4_CENTER[0]], [positions.EQ_SLOT4_CENTER[1]], marker="x", markersize=10, linewidth=1, color="yellow")
plt.plot([positions.UZYJ_BUTELKE_CONFIRMATION_BTN[0]], [positions.UZYJ_BUTELKE_CONFIRMATION_BTN[1]], marker="x", markersize=20, linewidth=1, color="green")
plt.plot([GAME_VIEW_POLY_OFF_BTN_POS[0]], [GAME_VIEW_POLY_OFF_BTN_POS[1]], marker="x", markersize=20, linewidth=1, color="purple")

plt.show()
