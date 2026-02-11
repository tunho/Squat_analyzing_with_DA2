import os
import glob
from squat_analyzer import SquatAnalyzer, DepthEstimator

def main():
    analyzer = SquatAnalyzer()
    
    # Target directories
    dataset_roots = [
        "dataset/true",
        "dataset/false"
    ]
    
    total_videos = 0
    processed_count = 0

    # Count total files first
    all_files = []
    for root in dataset_roots:
        mp4_files = glob.glob(os.path.join(root, "*.mp4"))
        all_files.extend(mp4_files)
    
    total_videos = len(all_files)
    print(f"Total videos to process: {total_videos}")

    for video_path in sorted(all_files):
        processed_count += 1
        
        # Construct output path
        # dataset/true/true_14.mp4 -> results/true/true_14.mp4
        
        # Split path parts
        parts = os.path.normpath(video_path).split(os.sep)
        # Assuming parts are ['dataset', 'true', 'filename.mp4']
        # We want results/true/filename.mp4
        
        if len(parts) >= 3:
            category = parts[-2]
            filename = parts[-1]
            out_dir = os.path.join("results", category)
        else:
            # Fallback
            category = "unknown"
            filename = os.path.basename(video_path)
            out_dir = os.path.join("results", category)
            
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filename)
        
        print(f"[{processed_count}/{total_videos}] Processing {video_path} -> {output_path} ...")
        
        # Run process (headless)
        # We need to reset analyzer state for each video!
        # Oh, SquatAnalyzer keeps state (self.counter, self.state).
        # We should create a new analyzer instance or reset it.
        # SquatAnalyzer.__init__ loads model which is heavy.
        # It's better to add a reset() method or manually reset variables.
        # Let's check SquatAnalyzer members.
        
        analyzer.reset()
        
        # We need to determine output path after processing to know the category
        # First save to a temp path or reconstruct after process_video returns
        temp_output = "results/temp_processing.mp4"
        count, had_issue, issue_list, issue_states = analyzer.process_video(video_path, temp_output, show_window=False)
        
        # [NEW] 2-Pass Re-Evaluation (Global Max Reference)
        # This re-runs the logic using the best body proportions found in the ENTIRE video.
        final_count, final_issues = analyzer.finalize_analysis()
        
        # Update results with 2-Pass conclusion
        count = final_count
        issue_list = final_issues
        # Note: had_issue logic might need update based on issue_list, but current logic handles count well.
        if len(issue_list) > 0: had_issue = True
        else: had_issue = False
        
        # Categorize
        if had_issue:
            # Determine state string
            if not issue_states:
                state_str = "UNKNOWN"
            elif len(issue_states) == 1:
                state_str = issue_states[0]
            else:
                state_str = "S2_AND_S3"

            # Determine issue subfolder
            if len(issue_list) == 1:
                issue_name = issue_list[0]
            else:
                issue_name = "COMPLEX"
                
            if count == 0:
                sub = f"issue_no_count/{state_str}/{issue_name}"
            else:
                sub = f"issue_with_count/{state_str}/{issue_name}"
        else:
            if count > 0:
                sub = "perfect_with_count"
            else:
                sub = "perfect_no_count"
            
        filename = os.path.basename(video_path)
        # Check environment variable or argument to switch between loose/strict output?
        # For now, let's just stick to results_loose as requested, I'll edit it later for strict.
        final_dir = os.path.join("results_strict", sub)
        os.makedirs(final_dir, exist_ok=True)
        final_output = os.path.join(final_dir, filename)
        
        # Move temp to final
        if os.path.exists(temp_output):
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(temp_output, final_output)
            
        print(f"Result: Count={count}, Issue={had_issue} (Types: {issue_list}, States: {issue_states}) -> {final_output}")

if __name__ == "__main__":
    main()
