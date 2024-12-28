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
        "hmls_01.ply",
        "hmls_02.ply",
        "hmls_03.ply",
        "hmls_04.ply",
        "hmls_05.ply",
        "hmls_06.ply",
        "hmls_07.ply",
        "hmls_08.ply",
        "hmls_09.ply",
        "hmls_10.ply",
        "hmls_11.ply",
        "hmls_12.ply",
        "hmls_13.ply",
        "hmls_14.ply",
        "hmls_16.ply",
        "hmls_17.ply",
        "hmls_18.ply",
        "hmls_19.ply",
        "hmls_20.ply",
        "hmls_21.ply",
        "hmls_22.ply",
        "hmls_23.ply",
        "hmls_24.ply",
        "hmls_25.ply",
        "hmls_26.ply",
        "hmls_27.ply",
        "sncf_01.ply",
        "sncf_02.ply",
        "sncf_03.ply",
        "sncf_04.ply",
        "sncf_05.ply",
        "sncf_06.ply",
        "sncf_07.ply",
        "sncf_08.ply",
        "sncf_09.ply",
        "sncf_10.ply",
        "sncf_11.ply",
        "sncf_12.ply",
        "sncf_13.ply",
        "sncf_14.ply",
        "sncf_15.ply",
    ],
    'val' : [
    ],
    'test' : [        
        "hmls_28.ply",
        "hmls_29.ply",
        "sncf_16.ply"
    ],
}