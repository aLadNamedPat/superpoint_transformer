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
    "class_8",
    "class_9",
    "class_10",
    # "void"
]

CLASS_COLORS = [
    [70, 130, 180],    # class_0
    [220, 20, 60],     # class_1
    [119, 11, 32],     # class_2
    [0, 0, 142],       # class_3
    [0, 60, 100],      # class_4
    [0, 80, 100],      # class_5
    [0, 0, 230],       # class_6
    [106, 0, 228],     # class_7
    [255, 255, 255],   # class_8
    [0, 255, 0],        # void
    [255, 0, 0]
]

TRACK_NUM_CLASSES = 10

STUFF_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ID2TRAINID = np.asarray([10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

TILES = {
    'train' : [
        "hmls_01",
        "hmls_02",
        "hmls_03",
        "hmls_04",
        "hmls_05",
        "hmls_06",
        "hmls_07",
        "hmls_08",
        "hmls_09",
        "hmls_10",
        "hmls_11",
        "hmls_12",
        "hmls_13",
        "hmls_14",
        "hmls_16",
        "hmls_17",
        "hmls_18",
        "hmls_19",
        "hmls_20",
        "hmls_21",
        "hmls_22",
        "hmls_23",
        "hmls_24",
        "hmls_25",
        "hmls_26",
        "hmls_27",
        "sncf_01",
        "sncf_02",
        "sncf_03",
        "sncf_04",
        "sncf_05",
        "sncf_06",
        "sncf_07",
        "sncf_08",
        "sncf_09",
        "sncf_10",
        "sncf_11",
        "sncf_12",
        "sncf_13",
        "sncf_14",
        "sncf_15",
    ],
    'val' : [
    ],
    'test' : [        
        "hmls_28",
        "hmls_29",
        "sncf_16"
    ],
}