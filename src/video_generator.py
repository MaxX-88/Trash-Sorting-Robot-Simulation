import cv2
import os
import glob
import numpy as np
from pathlib import Path

def create_video_from_frames(frames_dir, output_video_path, fps=60, frame_pattern="frame_*.jpg"):
    """
    Create a video from a directory of frames.
    
    Args:
        frames_dir (str): Directory containing the frame images
        output_video_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        frame_pattern (str): Pattern to match frame files (e.g., "frame_*.jpg")
    
    Returns:
        bool: True if video was created successfully, False otherwise
    """
    # Get all frame files matching the pattern
    frame_files = glob.glob(os.path.join(frames_dir, frame_pattern))
    
    if not frame_files:
        print(f"No frames found in {frames_dir} with pattern {frame_pattern}")
        return False
    
    # Sort files numerically (assuming they have frame numbers)
    frame_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_files[0]}")
        return False
    
    height, width, channels = first_frame.shape
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(frame_files)} frames...")
    print(f"Output: {output_video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    # Write each frame to the video
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
            if (i + 1) % 100 == 0:  # Progress update every 100 frames
                print(f"Processed {i + 1}/{len(frame_files)} frames")
        else:
            print(f"Warning: Could not read frame {frame_file}")
    
    # Release everything
    out.release()
    print(f"Video saved successfully: {output_video_path}")
    return True

def create_videos_from_simulation(project_root, fps=30, config=None):
    """
    Create videos from enabled camera frames.
    Creates a new Run folder for each simulation run.
    
    Args:
        project_root (str): Root directory of the project
        fps (int): Frames per second for output videos
        config: Configuration object with camera enable flags
    
    Returns:
        dict: Dictionary with camera names as keys and video paths as values
    """
    frames_base_dir = os.path.join(project_root, "frames")
    videos_base_dir = os.path.join(project_root, "videos")
    
    # Find the next run number
    run_number = 1
    while os.path.exists(os.path.join(videos_base_dir, f"Run{run_number}")):
        run_number += 1
    
    # Create the run-specific directory
    run_dir = os.path.join(videos_base_dir, f"Run{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Creating videos for Run{run_number}...")
    
    # Determine which cameras are enabled
    cameras = []
    if config is None or getattr(config, 'enable_perspective_frames', True):
        cameras.append("perspective")
    if config is None or getattr(config, 'enable_top_camera', False):
        cameras.append("top")
    if config is None or getattr(config, 'enable_detection_frames', True):
        cameras.append("detection")
    
    created_videos = {}
    
    for camera in cameras:
        frames_dir = os.path.join(frames_base_dir, camera)
        if os.path.exists(frames_dir) and os.listdir(frames_dir):
            output_video = os.path.join(run_dir, f"{camera}_simulation.mp4")
            success = create_video_from_frames(frames_dir, output_video, fps)
            if success:
                created_videos[camera] = output_video
        else:
            print(f"No frames found for {camera} camera in {frames_dir}")
    
    return created_videos

def cleanup_frames(project_root):
    """
    Clean up frame directories after video creation.
    
    Args:
        project_root (str): Root directory of the project
    """
    frames_base_dir = os.path.join(project_root, "frames")
    cameras = ["perspective", "top", "detection"]
    
    for camera in cameras:
        frames_dir = os.path.join(frames_base_dir, camera)
        if os.path.exists(frames_dir):
            frame_files = glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
            for frame_file in frame_files:
                os.remove(frame_file)
            print(f"Cleaned up {len(frame_files)} frames from {camera} camera")

if __name__ == "__main__":
    # Test the video generator
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print("Creating videos from simulation frames...")
    videos = create_videos_from_simulation(project_root, fps=30)
    
    if videos:
        print("\nVideos created:")
        for camera, video_path in videos.items():
            print(f"  {camera}: {video_path}")
    else:
        print("No videos were created. Make sure frames exist in the frames directory.")
