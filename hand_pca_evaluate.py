import cv2
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from mediapipe.framework.formats import landmark_pb2
import pickle

def extract_hand_features(video_path):
    """Extract hand landmarks from video with proper frame handling."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    # First pass: count frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Pre-allocate arrays for ALL frames
    n_landmarks = 21 * 3  # 21 landmarks, 3 coordinates each
    all_landmarks = np.zeros((total_frames, n_landmarks))  # Store all frames
    frame_data = [None] * total_frames  # Store MediaPipe results for visualization
    detection_mask = np.zeros(total_frames, dtype=bool)  # Track which frames had detections
    
    # Second pass: process frames
    cap = cv2.VideoCapture(video_path)
    print(f"\nProcessing {total_frames} frames...")
    
    last_valid_landmarks = None
    
    for frame_idx in range(total_frames):
        success, image = cap.read()
        if not success:
            break
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Take the first hand only for consistency
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_array = []
            for landmark in hand_landmarks.landmark:
                landmarks_array.extend([landmark.x, landmark.y, landmark.z])
            
            all_landmarks[frame_idx] = landmarks_array
            frame_data[frame_idx] = (frame_idx, hand_landmarks)
            detection_mask[frame_idx] = True
            last_valid_landmarks = landmarks_array
            
        elif last_valid_landmarks is not None:
            # Use last valid detection for missing frames
            all_landmarks[frame_idx] = last_valid_landmarks
            
            # Create visualization data from last valid landmarks
            last_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            for i in range(0, len(last_valid_landmarks), 3):
                landmark = landmark_pb2.NormalizedLandmark()
                landmark.x = last_valid_landmarks[i]
                landmark.y = last_valid_landmarks[i + 1]
                landmark.z = last_valid_landmarks[i + 2]
                last_landmarks_proto.landmark.append(landmark)
            
            frame_data[frame_idx] = (frame_idx, last_landmarks_proto)
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    
    # Remove trailing None entries if video ended early
    actual_frames = frame_idx + 1
    all_landmarks = all_landmarks[:actual_frames]
    frame_data = frame_data[:actual_frames]
    detection_mask = detection_mask[:actual_frames]
    
    # Check if we have any detections
    if not detection_mask.any():
        raise ValueError("No hand landmarks detected in the video!")
    
    print(f"\nDetection Summary:")
    print(f"Total frames: {actual_frames}")
    print(f"Frames with detections: {detection_mask.sum()}")
    print(f"Detection rate: {detection_mask.sum()/actual_frames*100:.1f}%")
    
    # Check for large gaps in detection
    gap_starts = np.where(np.diff(detection_mask.astype(int)) == -1)[0]
    gap_ends = np.where(np.diff(detection_mask.astype(int)) == 1)[0]
    
    if len(gap_starts) > 0:
        if len(gap_ends) < len(gap_starts):  # Gap continues to end of video
            gap_ends = np.append(gap_ends, len(detection_mask)-1)
        
        gaps = gap_ends - gap_starts
        large_gaps = gaps[gaps > 30]  # Gaps larger than 30 frames
        
        if len(large_gaps) > 0:
            print(f"\nWarning: Found {len(large_gaps)} large gaps in detection")
            print(f"Largest gap: {large_gaps.max()} frames")
            print(f"Average gap: {large_gaps.mean():.1f} frames")
            
            # Print the frame ranges of large gaps
            print("\nLarge gaps at frames:")
            for start, end, size in zip(gap_starts[gaps > 30], gap_ends[gaps > 30], large_gaps):
                print(f"  Frames {start} to {end} (gap of {size} frames)")
    
    return all_landmarks, frame_data, detection_mask

def compute_reconstruction_error(original, reconstructed):
    """Compute MSE between original and reconstructed data."""
    return np.mean((original - reconstructed) ** 2)

def create_comparison_video(video_path, output_path, frame_data, original_data, reconstructed_data, 
                          scaler, detection_mask, is_train=True):
    """Create video comparing original and PCA reconstruction."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info for {output_path}:")
    print(f"Original dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Detected frames: {np.sum(detection_mask)}")
    
    # Swap width and height for rotation
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (height * 2, width))
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    ORANGE = (0, 165, 255)
    
    def rotate_landmarks(landmarks_proto):
        """Rotate landmarks 90 degrees counterclockwise."""
        rotated = landmark_pb2.NormalizedLandmarkList()
        for landmark in landmarks_proto.landmark:
            new_landmark = landmark_pb2.NormalizedLandmark()
            # For 90° counterclockwise: x' = y, y' = 1-x
            new_landmark.x = landmark.y
            new_landmark.y = 1 - landmark.x
            new_landmark.z = landmark.z
            rotated.landmark.append(new_landmark)
        return rotated
    
    # Pre-process frames to ensure we have all needed frames
    frames = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        frames.append(image)
    cap.release()
    
    actual_frames = len(frames)
    if actual_frames != total_frames:
        print(f"Warning: Actual frames ({actual_frames}) differs from reported total ({total_frames})")
    
    processed_frames = 0
    last_frame_idx = -1
    
    for frame_idx in range(min(len(frames), len(frame_data))):
        # Skip if not detected
        if not detection_mask[frame_idx]:
            continue
            
        # Debug frame timing
        if last_frame_idx != -1 and frame_idx - last_frame_idx > 1:
            print(f"Gap in frames: {last_frame_idx} -> {frame_idx} ({frame_idx - last_frame_idx} frames)")
        last_frame_idx = frame_idx
        
        # Rotate image 90 degrees counterclockwise
        image = cv2.rotate(frames[frame_idx], cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        comparison = np.zeros((width, height * 2, 3), dtype=np.uint8)
        comparison[:, :height] = image
        comparison[:, height:] = image
        
        # Function to add text with background
        def put_text_with_background(img, text, pos, font_scale=1.0, thickness=2, 
                                   text_color=WHITE, bg_color=BLACK, padding=8):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            cv2.rectangle(img, (x-padding, y-text_h-padding), 
                         (x+text_w+padding, y+padding), bg_color, -1)
            cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
        
        # Add labels with white background
        dataset = "TRAINING" if is_train else "TEST"
        title_color = GREEN if is_train else ORANGE
        
        # Title and frame info (larger font)
        put_text_with_background(comparison, f"{dataset} VIDEO", (20, 50), 1.5, 3, title_color)
        put_text_with_background(comparison, f"Frame: {frame_idx}/{total_frames}", (20, 100), 1.2, 2)
        
        if frame_data[frame_idx] is not None:
            # Get and rotate the original landmarks
            _, landmarks = frame_data[frame_idx]
            rotated_landmarks = rotate_landmarks(landmarks)
            
            # Create and rotate reconstructed landmarks
            reconstructed = reconstructed_data[frame_idx]
            reconstructed = scaler.inverse_transform([reconstructed])[0]
            
            reconstructed_landmarks = landmark_pb2.NormalizedLandmarkList()
            for i in range(21):
                landmark = landmark_pb2.NormalizedLandmark()
                landmark.x = reconstructed[i*3]
                landmark.y = reconstructed[i*3+1]
                landmark.z = reconstructed[i*3+2]
                reconstructed_landmarks.landmark.append(landmark)
            
            rotated_reconstructed = rotate_landmarks(reconstructed_landmarks)
            
            # Draw rotated landmarks
            mp_drawing.draw_landmarks(
                comparison[:, :height], rotated_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=GREEN, thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
            )
            
            mp_drawing.draw_landmarks(
                comparison[:, height:], rotated_reconstructed, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=RED, thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
            )
            
            # Add error info (larger font)
            error = compute_reconstruction_error(
                original_data[frame_idx], 
                reconstructed_data[frame_idx]
            )
            
            put_text_with_background(comparison, f"Error: {error:.6f}", (20, 150), 1.2, 2)
        
        # Labels for original vs reconstruction (larger font)
        put_text_with_background(comparison, "Original", (height//2 - 150, width - 40), 
                               1.2, 2, text_color=GREEN)
        put_text_with_background(comparison, "PCA Reconstruction", 
                               (height + height//2 - 200, width - 40), 1.2, 2, text_color=RED)
        
        out.write(comparison)
        processed_frames += 1
        
        if processed_frames % 100 == 0:
            print(f"Processed {processed_frames} detected frames...")
    
    out.release()
    print(f"\nVideo processing summary for {output_path}:")
    print(f"Total frames in video: {actual_frames}")
    print(f"Frames with detections: {processed_frames}")
    print(f"Detection rate: {processed_frames/actual_frames*100:.1f}%")
    if last_frame_idx != -1:
        print(f"Frame range: 0 -> {last_frame_idx}")

def plot_metrics(train_reconstructed, test_reconstructed, explained_variance_ratio, output_prefix):
    """Plot evaluation metrics with improved clarity."""
    plt.style.use('default')  # Use default style instead of seaborn
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Reconstruction Errors
    plt.subplot(1, 2, 1)
    train_errors = np.mean((train_reconstructed) ** 2, axis=1)
    test_errors = np.mean((test_reconstructed) ** 2, axis=1)
    
    plt.hist([train_errors, test_errors], label=['Training', 'Test'], 
             bins=30, alpha=0.7, color=['green', 'orange'])
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Explained Variance
    plt.subplot(1, 2, 2)
    cumulative_var = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_var, 
            'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Components')
    plt.grid(True, alpha=0.3)
    
    # Add 95% line
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    plt.text(1, 0.96, '95% Variance', color='r', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define videos with correct filenames
    train_videos = [
        os.path.join("data", "train", "IMG_9465-train.MOV"),
        os.path.join("data", "train", "IMG_9466-train.MOV"),
        os.path.join("data", "train", "IMG_9467-train.MOV")
    ]
    test_video = os.path.join("data", "test", "IMG_9462-test.MOV")
    
    # Check if PCA and scaler already exist
    pca_file = os.path.join("models", "hand_pca_model.pkl")
    scaler_file = os.path.join("models", "hand_scaler.pkl")
    
    if os.path.exists(pca_file) and os.path.exists(scaler_file):
        print("\n=== Loading existing PCA and scaler ===")
        with open(pca_file, 'rb') as f:
            pca = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded pre-trained models")
        
        # Still need to process test video
        print("\n=== Processing Test Video ===")
        print("Test video:", test_video)
        test_landmarks, test_frame_data, test_mask = extract_hand_features(test_video)
        print(f"Test samples: {len(test_landmarks)}")
        
        # Transform test data
        test_landmarks_scaled = scaler.transform(test_landmarks)
        test_pca = pca.transform(test_landmarks_scaled)
        test_reconstructed = pca.inverse_transform(test_pca)
        
    else:
        print("\n=== Processing Training Videos ===")
        # Process all training videos
        train_landmarks_all = []
        train_frame_data_all = []
        train_mask_all = []
        
        for video in train_videos:
            print(f"\nProcessing training video: {video}")
            landmarks, frame_data, mask = extract_hand_features(video)
            train_landmarks_all.append(landmarks)
            train_frame_data_all.append(frame_data)
            train_mask_all.append(mask)
        
        # Combine all training data
        train_landmarks = np.vstack(train_landmarks_all)
        print(f"\nTotal training samples: {len(train_landmarks)}")
        
        print("\n=== Processing Test Video ===")
        print("Test video:", test_video)
        test_landmarks, test_frame_data, test_mask = extract_hand_features(test_video)
        print(f"Test samples: {len(test_landmarks)}")
        
        # Standardization
        print("\n=== Standardizing Data ===")
        scaler = StandardScaler()
        train_landmarks_scaled = scaler.fit_transform(train_landmarks)
        test_landmarks_scaled = scaler.transform(test_landmarks)
        
        # PCA
        print("\n=== Fitting PCA ===")
        n_components = 10  # Using 10 components to capture 99.42% variance
        pca = PCA(n_components=n_components)
        
        # Fit on combined training data
        train_pca = pca.fit_transform(train_landmarks_scaled)
        train_reconstructed = pca.inverse_transform(train_pca)
        
        # Transform test data
        test_pca = pca.transform(test_landmarks_scaled)
        test_reconstructed = pca.inverse_transform(test_pca)
        
        # Save PCA and scaler
        print("\n=== Saving PCA and scaler ===")
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved models to {pca_file} and {scaler_file}")
        
        # Split training reconstructions back to individual videos
        start_idx = 0
        for i, video in enumerate(train_videos):
            n_frames = len(train_landmarks_all[i])
            end_idx = start_idx + n_frames
            
            video_reconstructed = train_reconstructed[start_idx:end_idx]
            video_original = train_landmarks_scaled[start_idx:end_idx]
            
            # Create comparison video for each training video
            output_name = os.path.join("outputs", "videos", f"train_{i+1}_comparison.mp4")
            print(f"\nCreating {output_name}...")
            create_comparison_video(video, output_name, 
                                  train_frame_data_all[i], 
                                  video_original, 
                                  video_reconstructed,
                                  scaler, train_mask_all[i], is_train=True)
            
            start_idx = end_idx
    
    # Create test video
    test_output = os.path.join("outputs", "videos", "test_comparison.mp4")
    print(f"\nCreating {test_output}...")
    create_comparison_video(test_video, test_output, 
                          test_frame_data, test_landmarks_scaled, test_reconstructed, 
                          scaler, test_mask, is_train=False)
    
    # Compute metrics
    print("\n=== Computing Metrics ===")
    print("\nPCA Components Analysis:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        cumulative += var
        print(f"PC{i}: {var:.4f} ({cumulative:.4f} cumulative variance explained)")
    print(f"\nTotal variance explained by {pca.n_components_} components: {cumulative:.4%}")
    
    # Test errors
    test_errors = [compute_reconstruction_error(orig, rec) 
                  for orig, rec in zip(test_landmarks_scaled, test_reconstructed)]
    
    print("\nTest Video Metrics:")
    print(f"Mean reconstruction error: {np.mean(test_errors):.6f}")
    print(f"Std reconstruction error: {np.std(test_errors):.6f}")
    
    # Plot metrics
    metrics_output = os.path.join("outputs", "metrics", "hand_pca_metrics.png")
    plot_metrics(test_reconstructed, test_reconstructed, pca.explained_variance_ratio_, 
                os.path.splitext(metrics_output)[0])
    
    print("\nOutputs:")
    print(f"1. {test_output} - Test video comparison")
    print(f"2. {metrics_output} - Error distribution and explained variance plots")

if __name__ == "__main__":
    import pickle  # Add this at the top of the file
    main() 