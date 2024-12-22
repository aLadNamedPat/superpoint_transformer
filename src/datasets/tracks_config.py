import numpy as np


CLASS_NAMES = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "void"
]

CLASS_COLORS = [
    [70, 130, 180],   # class_0
    [220, 20, 60],    # class_1
    [119, 11, 32],    # class_2
    [0, 0, 142],      # class_3
    [0, 60, 100],     # class_4
    [0, 80, 100],     # class_5
    [0, 0, 230],      # class_6
    [106, 0, 228],    # class_7
    [255, 255, 255]
]

TRACK_NUM_CLASSES = 8

STUFF_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])

TILES = {
    'train' : [
    "hmls_01_0_1", 
    "hmls_01_0_2",
    "hmls_01_0_3",
    "hmls_01_0_4",
    "hmls_01_0_5",
    "hmls_01_0_6",
    "hmls_02_0_1",
    "hmls_02_0_2",
    "hmls_02_0_3",
    "hmls_03_0_1",
    "hmls_03_0_2",
    "hmls_03_0_3",
    "hmls_03_0_4",
    "hmls_04_0_1",
    "hmls_04_0_2",
    "hmls_04_0_3",
    "hmls_04_0_4",
    "hmls_04_0_5",
    "hmls_05_0_1",
    "hmls_05_0_2",
    "hmls_05_0_3",
    "hmls_05_0_4",
    "hmls_05_0_5",
    "hmls_06_0_1",
    "hmls_06_0_2",
    "hmls_06_0_3",
    "hmls_06_0_4",
    "hmls_07_0_1",
    "hmls_07_0_2",
    "hmls_07_0_3",
    "hmls_07_0_4",
    "hmls_07_0_5",
    "hmls_08_0_1",
    "hmls_08_0_2",
    "hmls_08_0_3",
    "hmls_08_0_4",
    # "hmls_08_0_5",
    ],
    'val' : [
    ],
    'test' : [],
}

