import pickle
import argparse
import os

from squat_analyzer_knee import SquatAnalyzer

def extract_to_pickle(video_path, output_pkl):
    print(f"Loading AI Model and analyzing video: {video_path}")
    analyzer = SquatAnalyzer()
    analyzer.process_video(video_path, show_window=False)
    _, _, params = analyzer.finalize_analysis()
    
    # We need to save: history_data, base_k, unstable_radii (stable_radii is basically standard)
    history = analyzer.history_data
    
    obs_max_thigh = max(f['thigh_len_2d'] for f in history)
    obs_max_shank = max(f['shank_len_2d'] for f in history)
    final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)
    final_shank_len = final_thigh_len / 1.2
    stable_radii = analyzer.corrector.get_radii(final_thigh_len, final_shank_len)
    
    data = {
        'history_data': history,
        'base_k': params.get('base_k', 1.0),
        'final_thigh_len': final_thigh_len,
        'final_shank_len': final_shank_len,
        'stable_radii': stable_radii
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"\n[Success] All AI data and frames extracted to '{output_pkl}'!")
    print(f"File size: {os.path.getsize(output_pkl) / (1024*1024):.1f} MB")
    print("Now you only need to download this .pkl file and lightweight_viewer.py to your laptop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, nargs='?', default='dataset/true/true_5.mp4')
    parser.add_argument('--out', type=str, default='squat_data.pkl')
    args = parser.parse_args()
    
    extract_to_pickle(args.video_path, args.out)
