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
    # [0, 255, 0]        # void
]

TRACK_NUM_CLASSES = 8

STUFF_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7])

TILES = {
    'train' : [
        "hmls_29_0_4", 
        "hmls_07_0_4", 
        "hmls_22_0_1", 
        "hmls_06_0_2", 
        "hmls_16_0_4", 
        "hmls_09_0_4", 
        "hmls_10_0_5", 
        "hmls_17_0_1", 
        "hmls_20_0_3", 
        "hmls_26_0_6", 
        "hmls_27_0_3", 
        "hmls_28_0_5", 
        "hmls_22_0_3", 
        "hmls_23_0_5", 
        "hmls_11_0_4", 
        "hmls_29_0_2", 
        "hmls_29_0_1", 
        "hmls_28_0_4", 
        "hmls_28_0_1", 
        "hmls_12_0_1", 
        "hmls_07_0_2", 
        "hmls_24_0_3", 
        "hmls_11_0_2", 
        "hmls_11_0_5", 
        "hmls_07_0_1", 
        "hmls_17_0_4", 
        "hmls_25_0_4", 
        "hmls_12_0_4", 
        "hmls_28_0_2", 
        "hmls_17_0_5", 
        "hmls_08_0_3", 
        "hmls_18_0_4", 
        "hmls_26_0_1", 
        "hmls_02_0_1", 
        "hmls_06_0_4", 
        "hmls_23_0_3", 
        "hmls_25_0_2", 
        "hmls_03_0_4", 
        "hmls_01_0_6", 
        "hmls_01_0_2", 
        "hmls_17_0_2", 
        "hmls_21_0_4", 
        "hmls_10_0_1", 
        "hmls_11_0_3", 
        "hmls_08_0_5", 
        "hmls_16_0_5", 
        "hmls_09_0_5", 
        "hmls_12_0_7", 
        "hmls_15_0_2", 
        "hmls_25_0_1", 
        "hmls_14_0_2", 
        "hmls_18_0_1", 
        "hmls_13_0_2", 
        "hmls_05_0_3", 
        "hmls_21_0_2", 
        "hmls_13_0_4", 
        "hmls_14_0_3", 
        "hmls_16_0_3", 
        "hmls_19_0_4", 
        "hmls_09_0_3", 
        "hmls_08_0_1", 
        "hmls_03_0_3", 
        "hmls_24_0_5", 
        "hmls_04_0_1", 
        "hmls_25_0_5", 
        "hmls_10_0_3", 
        "hmls_24_0_1", 
        "hmls_22_0_2", 
        "hmls_15_0_1", 
        "hmls_08_0_4", 
        "hmls_04_0_5", 
        "hmls_21_0_3", 
        "hmls_14_0_6", 
        "hmls_01_0_5", 
        "hmls_27_0_4", 
        "hmls_10_0_2", 
        "hmls_04_0_2", 
        "hmls_11_0_6", 
        "hmls_24_0_4", 
        "hmls_08_0_2", 
        "hmls_13_0_1", 
        "hmls_07_0_5", 
        "hmls_14_0_8", 
        "hmls_14_0_4", 
        "hmls_11_0_1", 
        "hmls_25_0_3", 
        "hmls_01_0_3", 
        "hmls_05_0_2", 
        "hmls_23_0_1", 
        "hmls_28_0_3", 
        "hmls_01_0_1", 
        "hmls_18_0_3", 
        "hmls_18_0_2", 
        "hmls_19_0_2", 
        "hmls_12_0_5", 
        "hmls_20_0_2", 
        "hmls_15_0_3", 
        "hmls_23_0_2", 
        "hmls_26_0_4", 
        "hmls_06_0_3", 
        "hmls_07_0_3", 
        "hmls_26_0_5", 
        "hmls_13_0_5", 
        "hmls_14_0_5", 
        "hmls_16_0_1", 
        "hmls_24_0_2", 
        "hmls_06_0_1", 
        "hmls_03_0_2", 
        "hmls_09_0_2", 
        "hmls_08_0_6", 
        "hmls_17_0_3", 
        "hmls_26_0_3", 
        "hmls_23_0_4", 
        "hmls_15_0_5", 
        "hmls_12_0_2", 
        "hmls_19_0_1", 
        "hmls_22_0_4", 
        "hmls_14_0_7", 
        "hmls_27_0_2", 
        "hmls_05_0_1", 
        "hmls_27_0_5", 
        "hmls_19_0_3", 
        "hmls_01_0_4", 
        "hmls_02_0_2", 
        "hmls_27_0_1", 
        "hmls_04_0_4", 
        "hmls_26_0_2", 
        "hmls_20_0_4", 
        "hmls_12_0_6", 
        "hmls_29_0_3", 
        "hmls_05_0_5", 
        "hmls_15_0_4", 
        "hmls_05_0_4", 
        "hmls_13_0_3", 
        "hmls_19_0_5", 
        "hmls_21_0_1", 
        "hmls_03_0_1", 
        "hmls_09_0_1", 
        "hmls_14_0_1", 
        "hmls_18_0_5", 
        "hmls_09_0_6", 
        "hmls_16_0_2", 
        "hmls_12_0_3", 
        "hmls_02_0_3",
        "sncf_04_0_9", 
        "sncf_08_0_11", 
        "sncf_13_0_4", 
        "sncf_10_0_10", 
        "sncf_14_0_2", 
        "sncf_04_0_12", 
        "sncf_06_0_14", 
        "sncf_08_0_2", 
        "sncf_05_0_2", 
        "sncf_14_0_5", 
        "sncf_01_0_10", 
        "sncf_10_0_4", 
        "sncf_05_0_12", 
        "sncf_01_0_4", 
        "sncf_11_0_9", 
        "sncf_03_0_1", 
        "sncf_11_0_8", 
        "sncf_07_0_10", 
        "sncf_07_0_14", 
        "sncf_11_0_11", 
        "sncf_16_0_5", 
        "sncf_14_0_4", 
        "sncf_09_0_12", 
        "sncf_15_0_8", 
        "sncf_03_0_9", 
        "sncf_16_0_8", 
        "sncf_02_0_6", 
        "sncf_12_0_2", 
        "sncf_10_0_7", 
        "sncf_03_0_4", 
        "sncf_11_0_12", 
        "sncf_06_0_2", 
        "sncf_15_0_7", 
        "sncf_02_0_8", 
        "sncf_05_0_1", 
        "sncf_10_0_3", 
        "sncf_07_0_12", 
        "sncf_02_0_5", 
        "sncf_16_0_7", 
        "sncf_16_0_6", 
        "sncf_11_0_4", 
        "sncf_09_0_7", 
        "sncf_09_0_11", 
        "sncf_11_0_10", 
        "sncf_15_0_5", 
        "sncf_15_0_3", 
        "sncf_15_0_11", 
        "sncf_05_0_13", 
        "sncf_02_0_4", 
        "sncf_03_0_5", 
        "sncf_15_0_12", 
        "sncf_06_0_4", 
        "sncf_07_0_8", 
        "sncf_04_0_5", 
        "sncf_02_0_9", 
        "sncf_09_0_9", 
        "sncf_08_0_5", 
        "sncf_14_0_1", 
        "sncf_16_0_1", 
        "sncf_04_0_7", 
        "sncf_01_0_8", 
        "sncf_13_0_6", 
        "sncf_02_0_10", 
        "sncf_11_0_1", 
        "sncf_09_0_3", 
        "sncf_07_0_11", 
        "sncf_12_0_3", 
        "sncf_11_0_7", 
        "sncf_13_0_10", 
        "sncf_01_0_2", 
        "sncf_02_0_2", 
        "sncf_08_0_9", 
        "sncf_05_0_10", 
        "sncf_06_0_8", 
        "sncf_12_0_5", 
        "sncf_08_0_4", 
        "sncf_03_0_11", 
        "sncf_06_0_11", 
        "sncf_07_0_7", 
        "sncf_13_0_1", 
        "sncf_06_0_15", 
        "sncf_14_0_8", 
        "sncf_13_0_2", 
        "sncf_05_0_11", 
        "sncf_09_0_5", 
        "sncf_06_0_7", 
        "sncf_10_0_2", 
        "sncf_04_0_1", 
        "sncf_15_0_4", 
        "sncf_14_0_3", 
        "sncf_06_0_9", 
        "sncf_14_0_10", 
        "sncf_07_0_6", 
        "sncf_16_0_9", 
        "sncf_04_0_6", 
        "sncf_07_0_3", 
        "sncf_01_0_7", 
        "sncf_05_0_6", 
        "sncf_08_0_10", 
        "sncf_09_0_10", 
        "sncf_02_0_1", 
        "sncf_09_0_8", 
        "sncf_05_0_9", 
        "sncf_04_0_2", 
        "sncf_11_0_6", 
        "sncf_13_0_5", 
        "sncf_08_0_8", 
        "sncf_06_0_5", 
        "sncf_07_0_2", 
        "sncf_06_0_12", 
        "sncf_14_0_9", 
        "sncf_13_0_11", 
        "sncf_12_0_6", 
        "sncf_12_0_10", 
        "sncf_15_0_9", 
        "sncf_04_0_11", 
        "sncf_01_0_6", 
        "sncf_11_0_2", 
        "sncf_05_0_4", 
        "sncf_08_0_7", 
        "sncf_09_0_4", 
        "sncf_02_0_7", 
        "sncf_12_0_4", 
        "sncf_02_0_3", 
        "sncf_03_0_7", 
        "sncf_09_0_1", 
        "sncf_05_0_7", 
        "sncf_10_0_8", 
        "sncf_16_0_2", 
        "sncf_16_0_4", 
        "sncf_01_0_5", 
        "sncf_15_0_10", 
        "sncf_11_0_5", 
        "sncf_04_0_4", 
        "sncf_13_0_3", 
        "sncf_09_0_6", 
        "sncf_06_0_13", 
        "sncf_03_0_2", 
        "sncf_01_0_3", 
        "sncf_05_0_3", 
        "sncf_06_0_6", 
        "sncf_06_0_3", 
        "sncf_08_0_1", 
        "sncf_14_0_7", 
        "sncf_08_0_3", 
        "sncf_03_0_6", 
        "sncf_07_0_4", 
        "sncf_01_0_1", 
        "sncf_12_0_11", 
        "sncf_10_0_6", 
        "sncf_13_0_7", 
        "sncf_05_0_5", 
        "sncf_09_0_2", 
        "sncf_14_0_6", 
        "sncf_03_0_10", 
        "sncf_05_0_8", 
        "sncf_04_0_3", 
        "sncf_07_0_13", 
        "sncf_11_0_3", 
        "sncf_08_0_6", 
        "sncf_07_0_9", 
        "sncf_03_0_12", 
        "sncf_10_0_9", 
        "sncf_15_0_2", 
        "sncf_10_0_11", 
        "sncf_06_0_10", 
        "sncf_02_0_11", 
        "sncf_12_0_9", 
        "sncf_13_0_9", 
        "sncf_16_0_3", 
        "sncf_08_0_12", 
        "sncf_15_0_1", 
        "sncf_12_0_8", 
        "sncf_15_0_6", 
        "sncf_12_0_7", 
        "sncf_03_0_3", 
        "sncf_03_0_8", 
        "sncf_10_0_1", 
        "sncf_12_0_1", 
        "sncf_04_0_8", 
        "sncf_10_0_5", 
        "sncf_07_0_5", 
        "sncf_07_0_1", 
        "sncf_04_0_10"
    ],
    'val' : [
        "hmls_20_0_1", 
        "hmls_04_0_3", 
        # "hmls_10_0_4", 
        # "sncf_01_0_9", 
        "sncf_13_0_8", 
        "sncf_06_0_1",
    ],
    'test' : [],
}

