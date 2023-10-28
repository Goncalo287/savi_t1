# SAVI - Trabalho Prático 1


**Group 8**

GONÇALO IVAN RAMOS ANACLETO - 93394

CAROLINA OLIVEIRA FRANCISCO - 98303


## Introduction

A python program using OpenCV to detect and track specific people in the camera. Includes a gallery of saved templates and other features that improve usability.



## Features

- Face detection using Haar Cascade
- Face tracking using template matching
- Continuous tracking even when faces aren't detected
- Handles tracking of multiple people simultaneously
- Ability to save and load tracking templates
- Saved templates can be viewed in a separate window, updated in real-time
- Recognized faces are greeted using Google Text-to-Speech
- Audio files created with gTTS are stored, allowing offline use
- Unknown faces can be easily identified by clicking on them
- Auto-refresh when face tracking stops working as intended
- Tracking parameters can be adjusted in the UI while the program is running



## How to Use

- Clone this repository and run `main.py` using one of the following commands:
```console
$ ./main.py
$ python3 main.py
```

- When running the program, the camera will open and detect people's faces
- To create a new tracker, click on one of the detected faces
- Insert the name for the new tracker as requested to continue
- Tracked faces are highlighted with their names and added to the "database" window
- The tracker templates can be saved and loaded from files



## Key Bindings

| Key | Use |
| - | - |
| Q | Exit the program (ESC) |
| T | Create a tracker from a region (an alternative to mouse click) |
| R | Reset trackers to their original template |
| S | Save trackers to disk |
| L | Load trackers from disk |



## Dependencies

```console
$ sudo apt-get install python3-opencv
$ sudo apt-get install python3-tk
$ pip install gTTS
```

All other required libraries should be included with Ubuntu
