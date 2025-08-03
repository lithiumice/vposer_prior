
SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip", # 
    "spine1",
    "left_knee",
    "right_knee", # 5
    "spine2",
    "left_ankle", # 7
    "right_ankle", # 8
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",   # 15
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",    # 22
    "left_eye_smplhf",
    "right_eye_smplhf", # 24
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3", # 30
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3", # 36
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",   # 42
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",   # 48
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",   # 55
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",   # 60
    "left_small_toe", # 61
    "left_heel",  # 62
    "right_big_toe",  # 63
    "right_small_toe",    # 64
    "right_heel", # 65
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky", # 70
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",    # 75
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",    # 80
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1", # 85
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",   # 90
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1", # 95
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6", # 100
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",  # 105
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",  # 110
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",   # 116
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",    # 120
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom", # 125
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",    # 130
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",    # 135
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5", # 140
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1", # 144
]


# Skeletons for SMPLX joints
SMPLX_skeleton_connections = [
    [ 0, 1 ],
    [ 0, 2 ],
    [ 0, 3 ],
    [ 1, 4 ],
    [ 2, 5 ],
    [ 3, 6 ],
    [ 4, 7 ],
    [ 5, 8 ],
    [ 6, 9 ],
    [ 7, 10],
    [ 8, 11],
    [ 9, 12],
    [ 9, 13],
    [ 9, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    # [20, 25],   # left hand from my view
    # [25, 26],   # left_index
    # [26, 27],   
    # [27, 67],   # finger tips
    # [20, 28],
    # [28, 29],   # left_middle
    # [29, 30],   
    # [30, 68],
    # [20, 31],
    # [31, 32],   # left_pinky
    # [32, 33],
    # [33, 70],
    # [20, 34],
    # [34, 35],   # left_ring
    # [35, 36],
    # [36, 69],
    # [20, 37],
    # [37, 38],   # left_thumb
    # [38, 39],
    # [39, 66],   # finger tips
    # [21, 40],   # right hand from my view
    # [40, 41],   # right_index
    # [41, 42],   
    # [42, 72],   # finger tips
    # [21, 43],
    # [43, 44],   # right_middle
    # [44, 45],   
    # [45, 73],
    # [21, 46],   # right_pinky
    # [46, 47],
    # [47, 48],
    # [48, 75],
    # [21, 49],
    # [49, 50],   # right_ring
    # [50, 51],
    # [51, 74],
    # [21, 52],
    # [52, 53],   # right_thumb
    # [53, 54],
    # [54, 71],   # finger tips
    # [ 7, 60],
    # [60, 61],
    # [ 7, 62],
    # [ 8, 63],
    # [63, 64],
    # [ 8, 65],
]