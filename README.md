# Metin2 (MMORPG) Vision Bot

Automatic Bot for Massive Dungeons Passing based on Windows API, YoloV8 object detection, statistical methods from OpenCV, Tesseract-OCR and spaCy virtualised with Hyper-V (Win11+CUDA)

## Key Packages
- Keyboard and Mouse controll: `pynput`
- Custom Data Annotation: `Roboflow`
- Object detection: `ultralytics`, `OpenCV`
- Messages handling: `Tesseract-OCR`, `spaCy`

## Funcionalities
- Autonomic Dungeon Passing
- Autonomic Metin Stones Destroying among all of the channels
- Idle Exp

> All of those modes can handle boundary situations such as game crash or logging out.

## YoloV8
Ultralytics YOLOv8.0.196 🚀 Python-3.10.12 torch-2.1.0+cu121 CUDA:0 (Tesla T4, 15102MiB)

| Class | Images | Instances | Precision* | Recall* | mAP50* | mAP50-95* |
|-|:-:|-:|-:|-:|-:|-:|
| all | 31 | 36 | 0.993 | 0.939 | 0.974 | 0.828 |
| boss_gnoll_cpt | 31 | 11 | 0.983 | 0.818 | 0.931 | 0.739 |
| metin_polany | 31 | 12 | 0.996 | 1 | 0.995 | 0.867 |
| npc_straznik | 31 | 13 | 0.999 | 1 | 0.995 | 0.877 |

\* - Box level

<div style="width:20%; margin: auto;">
  <img width=80% src=https://github.com/jamnicki/metin2_vision_bot/assets/56606076/36d47e7a-ce6a-40ab-a71c-dcb0216e4333>
  <img width=50% src=https://github.com/jamnicki/metin2_vision_bot/assets/56606076/34e5c6af-0377-40f9-b923-cab9951ffb1a>
  <img width=80% src=https://github.com/jamnicki/metin2_vision_bot/assets/56606076/2316a2b4-4551-4572-8825-f70b270ffd51>
  <img width=60% src=https://github.com/jamnicki/metin2_vision_bot/assets/56606076/2b024e26-d2b6-4dfc-b757-8db8de789faf>
</div>

