import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import argparse
from tqdm import tqdm

# Set numpy print options
np.set_printoptions(threshold=100, precision=3)

def analyze_motion_data(data, verbose=True):
    """Analyze the motion data in detail."""
    # Get the main motion data
    motion_key = list(data.keys())[0]
    motion_data = data[motion_key]
    
    if verbose:
        print(f"Motion name: {motion_key}")
        print("\nData structure:")
        for key, value in motion_data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
    
    stats = {}
    
    # Analyze root translation
    root_trans = motion_data['root_trans_offset']
    stats['root_trans'] = {
        'min': root_trans.min(axis=0),
        'max': root_trans.max(axis=0),
        'mean': root_trans.mean(axis=0),
        'range': root_trans.max(axis=0) - root_trans.min(axis=0)
    }
    
    if verbose:
        print("\nRoot translation statistics:")
        print(f"  Min: {stats['root_trans']['min']}")
        print(f"  Max: {stats['root_trans']['max']}")
        print(f"  Mean: {stats['root_trans']['mean']}")
        print(f"  Range: {stats['root_trans']['range']}")
    
    # Analyze pose angles
    pose_aa = motion_data['pose_aa']
    stats['pose_aa'] = {
        'min': pose_aa.min(axis=(0,1)),
        'max': pose_aa.max(axis=(0,1)),
        'mean': pose_aa.mean(axis=(0,1)),
        'range': pose_aa.max(axis=(0,1)) - pose_aa.min(axis=(0,1))
    }
    
    if verbose:
        print("\nPose angles statistics:")
        print(f"  Min: {stats['pose_aa']['min']}")
        print(f"  Max: {stats['pose_aa']['max']}")
        print(f"  Mean: {stats['pose_aa']['mean']}")
        print(f"  Range: {stats['pose_aa']['range']}")
    
    # Analyze DOF values
    dof = motion_data['dof']
    stats['dof'] = {
        'min': dof.min(axis=0),
        'max': dof.max(axis=0),
        'mean': dof.mean(axis=0),
        'range': dof.max(axis=0) - dof.min(axis=0)
    }
    
    if verbose:
        print("\nDOF statistics:")
        print(f"  Min: {stats['dof']['min']}")
        print(f"  Max: {stats['dof']['max']}")
        print(f"  Mean: {stats['dof']['mean']}")
        print(f"  Range: {stats['dof']['range']}")
    
    # Analyze root rotation
    root_rot = motion_data['root_rot']
    stats['root_rot'] = {
        'min': root_rot.min(axis=0),
        'max': root_rot.max(axis=0),
        'mean': root_rot.mean(axis=0),
        'range': root_rot.max(axis=0) - root_rot.min(axis=0)
    }
    
    if verbose:
        print("\nRoot rotation (quaternion) statistics:")
        print(f"  Min: {stats['root_rot']['min']}")
        print(f"  Max: {stats['root_rot']['max']}")
        print(f"  Mean: {stats['root_rot']['mean']}")
        print(f"  Range: {stats['root_rot']['range']}")
    
    # Analyze SMPL joints
    smpl_joints = motion_data['smpl_joints']
    stats['smpl_joints'] = {
        'min': smpl_joints.min(axis=(0,1)),
        'max': smpl_joints.max(axis=(0,1)),
        'mean': smpl_joints.mean(axis=(0,1)),
        'range': smpl_joints.max(axis=(0,1)) - smpl_joints.min(axis=(0,1))
    }
    
    if verbose:
        print("\nSMPL joints statistics:")
        print(f"  Min: {stats['smpl_joints']['min']}")
        print(f"  Max: {stats['smpl_joints']['max']}")
        print(f"  Mean: {stats['smpl_joints']['mean']}")
        print(f"  Range: {stats['smpl_joints']['range']}")
    
    # Print frame count and duration
    fps = motion_data['fps']
    num_frames = root_trans.shape[0]
    duration = num_frames / fps
    
    stats['frames'] = {
        'count': num_frames,
        'duration': duration,
        'fps': fps
    }
    
    if verbose:
        print(f"\nMotion duration: {duration:.2f} seconds ({num_frames} frames at {fps} fps)")
        
        # Print first few frames of DOF values
        print("\nFirst 5 frames of DOF values:")
        for i in range(min(5, dof.shape[0])):
            print(f"Frame {i}: {dof[i]}")
    
    return motion_data, stats

def save_dof_to_txt(motion_data, file_path):
    """Save DOF data to a text file with the same name as the pickle file."""
    # Extract dof data
    dof_data = motion_data['dof']
    
    # Create output file path with the same name but .txt extension
    base_path = os.path.splitext(file_path)[0]
    txt_file_path = f"{base_path}.txt"
    
    # Save dof data to the text file
    np.savetxt(txt_file_path, dof_data, fmt='%.6f')
    
    print(f"\nSaved DOF data to: {txt_file_path}")

def analyze_directory(directory_path, save_txt=False):
    """Analyze all pickle files in a directory and compute aggregate statistics."""
    print(f"Analyzing all pickle files in: {directory_path}")
    
    # Get all pickle files in the directory
    pickle_files = glob.glob(os.path.join(directory_path, "*.pkl"))
    
    if not pickle_files:
        print("No pickle files found in the directory.")
        return
    
    print(f"Found {len(pickle_files)} pickle files.")
    
    # Initialize aggregate statistics variables
    all_stats = []
    total_frames = 0
    total_duration = 0
    frame_counts = []
    durations = []
    fps_values = []
    
    # Process each pickle file
    for file_path in tqdm(pickle_files, desc="Processing files"):
        try:
            # Load the data using joblib
            data = joblib.load(file_path)
            
            # Get the filename
            filename = os.path.basename(file_path)
            print(f"\n\nAnalyzing file: {filename}")
            
            # Analyze the motion data (without verbose output)
            motion_data, stats = analyze_motion_data(data, verbose=False)
            
            # Save DOF data to text file if requested
            if save_txt:
                save_dof_to_txt(motion_data, file_path)
            
            # Collect statistics
            all_stats.append(stats)
            total_frames += stats['frames']['count']
            total_duration += stats['frames']['duration']
            frame_counts.append(stats['frames']['count'])
            durations.append(stats['frames']['duration'])
            fps_values.append(stats['frames']['fps'])
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Calculate aggregate statistics
    avg_frames = total_frames / len(pickle_files)
    avg_duration = total_duration / len(pickle_files)
    avg_fps = sum(fps_values) / len(fps_values)
    
    # Convert to numpy arrays for easier statistics
    frame_counts = np.array(frame_counts)
    durations = np.array(durations)
    fps_values = np.array(fps_values)
    
    print("\n" + "="*50)
    print("AGGREGATE STATISTICS")
    print("="*50)
    print(f"Total number of pickle files: {len(pickle_files)}")
    print(f"Total frames across all files: {total_frames}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print("\nFrame statistics:")
    print(f"  Average frames per file: {avg_frames:.2f}")
    print(f"  Min frames: {frame_counts.min()}")
    print(f"  Max frames: {frame_counts.max()}")
    print(f"  Median frames: {np.median(frame_counts):.1f}")
    print(f"  Std dev frames: {np.std(frame_counts):.2f}")
    
    print("\nDuration statistics:")
    print(f"  Average duration per file: {avg_duration:.2f} seconds")
    print(f"  Min duration: {durations.min():.2f} seconds")
    print(f"  Max duration: {durations.max():.2f} seconds")
    print(f"  Median duration: {np.median(durations):.2f} seconds")
    print(f"  Std dev duration: {np.std(durations):.2f} seconds")
    
    print("\nFPS statistics:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Min FPS: {fps_values.min():.2f}")
    print(f"  Max FPS: {fps_values.max():.2f}")
    
    # Plot histogram of frame counts
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(frame_counts, bins=20)
    plt.title('Histogram of Frame Counts')
    plt.xlabel('Number of Frames')
    plt.ylabel('Number of Files')
    
    # Plot histogram of durations
    plt.subplot(1, 2, 2)
    plt.hist(durations, bins=20)
    plt.title('Histogram of Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Files')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(directory_path), "statistics")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "motion_statistics.png"))
    print(f"\nSaved statistics plot to: {os.path.join(output_dir, 'motion_statistics.png')}")
    
    # Show the plot
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze motion data from pickle files')
    parser.add_argument('--file', type=str, help='Path to a single pickle file to analyze')
    parser.add_argument('--dir', type=str, help='Path to a directory containing pickle files to analyze')
    parser.add_argument('--save_txt', action='store_true', help='Save DOF data to text files')
    
    args = parser.parse_args()
    
    # Determine what to analyze based on arguments
    if args.file:
        # Analyze a single file
        print(f"Loading pickle file: {args.file}")
        try:
            # Load the data using joblib
            data = joblib.load(args.file)
            print("Successfully loaded the data")
            
            # Analyze the motion data
            motion_data, _ = analyze_motion_data(data)
            
            # Save DOF data to text file if requested
            if args.save_txt:
                save_dof_to_txt(motion_data, args.file)
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.dir:
        # Analyze all files in a directory
        analyze_directory(args.dir, save_txt=args.save_txt)
    
    else:
        default_dir = "humanoidverse/data/motions/adam/Arm Movements"
        print(f"No file or directory specified. Using default directory: {default_dir}")
        analyze_directory(default_dir, save_txt=False)

if __name__ == "__main__":
    main()
