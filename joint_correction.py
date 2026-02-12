
import numpy as np

class JointCenterCorrector:
    """
    A unified interface for correcting surface keypoints (from MediaPipe + Depth Anything)
    to true joint centers (skeletal joints).
    
    This class supports a Strategy Pattern:
    1. 'geometric': (Current) Uses simple cylindrical offset based on limb lengths.
    2. 'skel': (Future) Can wrap a full SMPL/HMR mesh model.
    """
    
    def __init__(self, mode='geometric'):
        self.mode = mode
        self.initialized = True
        
        # [Geometric Parameters]
        # Ratios relative to Thigh Length (L_thigh) or Shank Length (L_shank)
        # These are approximate anthropometric averages for "Radius" (half-thickness)
        self.RATIO_HIP_RADIUS = 0.14   # Hip is thick (Glutes/Quads origin)
        self.RATIO_KNEE_RADIUS = 0.07  # Knee is thin (Patella + Condyles)
        self.RATIO_ANKLE_RADIUS = 0.05 # Ankle is very thin
        
        # Debug / Calibration
        self.last_offsets = {} 

    def correct(self, joints, meta_data):
        """
        Main entry point for correction.
        
        Args:
            joints (dict): {'hip': {'x', 'y', 'z', ...}, 'knee': ..., 'ankle': ...}
            meta_data (dict): Must contain 'thigh_len' and 'shank_len' (pixels or units)
                              Preferably 'max_thigh_len' for stability.
        
        Returns:
            dict: A new dictionary of corrected joints (Deep Copy-ish)
        """
        # Defensive copy to avoid mutating original surface landmarks if needed
        # But for performance we might just return new dicts for the points
        corrected_joints = {}
        for k, v in joints.items():
            corrected_joints[k] = v.copy()
            
        if self.mode == 'geometric':
            return self._apply_geometric_offset(corrected_joints, meta_data)
        elif self.mode == 'skel':
            return self._apply_skel_model(corrected_joints, meta_data)
        else:
            print(f"[JointCorrector] Unknown mode '{self.mode}', returning surface joints.")
            return corrected_joints

    def _apply_geometric_offset(self, joints, meta):
        """
        Implements the 'Cylindrical Offset Model'.
        Z_center = Z_surface + Radius (Assumes +Z is direction away from camera, so Center is deeper)
        """
        # 1. Get Reference Lengths (Stable)
        ref_thigh = meta.get('max_thigh_len', meta.get('thigh_len', 0))
        ref_shank = meta.get('max_shank_len', meta.get('shank_len', 0))
        
        # 2. Get Radii
        radii = self.get_radii(ref_thigh, ref_shank)
        r_hip = radii['hip']
        r_knee = radii['knee']
        r_ankle = radii['ankle']

        # 3. Apply Z-Push (Inwards)
        # We assume Z increases "away" from camera (Depth).
        # Surface is closer (smaller Z) than Bone (larger Z).
        # So Joint = Surface + Radius.
        
        joints['hip']['z'] += r_hip
        joints['knee']['z'] += r_knee
        joints['ankle']['z'] += r_ankle
        
        # Debug info store
        self.last_offsets = {
            'r_hip': r_hip,
            'r_knee': r_knee,
            'r_ankle': r_ankle
        }
        
        return joints

    def get_radii(self, ref_thigh, ref_shank):
        """
        Calculate radii based on reference limb lengths.
        Return dict: {'hip': r, 'knee': r, 'ankle': r}
        """
        if ref_thigh == 0: ref_thigh = 100 
        if ref_shank == 0: ref_shank = 100 

        return {
            'hip': ref_thigh * self.RATIO_HIP_RADIUS,
            'knee': ref_thigh * self.RATIO_KNEE_RADIUS,
            'ankle': ref_shank * self.RATIO_ANKLE_RADIUS
        }

    def _apply_skel_model(self, joints, meta):
        """
        Placeholder for Future Mesh-based correction.
        This would invoke a neural network (e.g. HMR, SPIN) to regressed 3D joints.
        """
        # TODO: Implement SKEL model inference here.
        # For now, just pass through.
        return joints
