import numpy as np

from minigrid.core.constants import (
    OBJECT_TO_IDX,
    IDX_TO_OBJECT,
    COLORS,
    COLOR_NAMES,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    STATE_TO_IDX,
    DIR_TO_VEC,
)

# direction
VEC_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_TO_NAME = {0: "right", 1: "down", 2: "left", 3: "up"}
IDX_TO_DIR = {0: "east", 1: "south", 2: "west", 3: "north"}

# object
OBJECT_TO_IDX["heavy_door"] = 11
OBJECT_TO_IDX["bridge"] = 12
OBJECT_TO_IDX["fireproof_shoes"] = 13
OBJECT_TO_IDX["hammer"] = 14
OBJECT_TO_IDX["passage"] = 15
OBJECT_TO_IDX["safe_lava"] = 16
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}


# color
COLORS["brown"] = np.array([119, 70, 20])
COLOR_TO_IDX["brown"] = 6

COLORS["turquoise"] = np.array([64, 224, 208])
COLOR_TO_IDX["turquoise"] = 7

COLORS["saffron"] = np.array([244, 196, 48])
COLOR_TO_IDX["saffron"] = 8

COLORS["indigo"] = np.array([75, 0, 130])
COLOR_TO_IDX["indigo"] = 9

COLORS["salmon"] = np.array([250, 128, 114])
COLOR_TO_IDX["salmon"] = 10

COLORS["lime"] = np.array([191, 255, 0])
COLOR_TO_IDX["lime"] = 11

IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
COLOR_NAMES = sorted(list(COLORS.keys()))



