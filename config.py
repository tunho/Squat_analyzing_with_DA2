# Configuration for Squat Analysis Project
# Central place for all adjustable parameters and thresholds.

import torch

# 1. Model Configuration
MODEL_CONFIG = {
    'DEPTH_MODEL_ID': "depth-anything/Depth-Anything-V2-Small-hf",
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'POSE_MODEL_COMPLEXITY': 1, # 0=Lite, 1=Full, 2=Heavy
    'MIN_DETECTION_CONFIDENCE': 0.5,
    'MIN_TRACKING_CONFIDENCE': 0.5
}

# 2. Squat Analysis Thresholds
SQUAT_THRESHOLDS = {
    'KNEE_ANGLE_STANDING': 170.0, # Angle considered as "Standing Up" (Extension)
    'KNEE_ANGLE_DOWN': 95.0,      # Angle required to count as "Bottom Position" (Flexion)
    'KNEE_ANGLE_UP': 165.0,       # Angle required to complete a rep (Return)
    'REQUIRED_FLEXION_RANGE': 60.0, # Minimum ROM (Range of Motion)
    
    'TRUNK_LEAN_LIMIT': 35.0,     # Maximum allowed trunk forward lean (degrees)
    'KNEE_VALGUS_LIMIT': 20.0     # Maximum allowed knee inward collapse (degrees)
}

# 3. Data Smoothing (Signal Processing)
FILTER_CONFIG = {
    'ENABLED': True,
    'WINDOW_LENGTH': 15,          # Must be odd. Larger = smoother but more lag.
    'POLY_ORDER': 3               # Polynomial order for Savitzky-Golay filter.
}

# 4. 3D Viewer Settings
VIEWER_CONFIG = {
    'AXIS_LIMIT': 400,            # 3D Plot size (mm/pixels equivalent)
    'ANIMATION_INTERVAL': 50,     # ms per frame (50ms = 20fps playback)
    'POINT_SIZE': 8,
    'LINE_WIDTH': 4,
    'DEFAULT_ELEV': 10,
    'DEFAULT_AZIM': -90
}

# 5. Output Settings
OUTPUT_CONFIG = {
    'VIDEO_FPS': 30.0,
    'DRAW_SCALE': 0.8,            # Scale factor for multiview drawing
    'DEFAULT_OUTPUT_FILENAME': "output_squat_analysis.mp4"
}
