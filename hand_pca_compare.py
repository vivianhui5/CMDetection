import cv2
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from mediapipe.framework.formats import landmark_pb2

def draw_hand_landmarks(image, landmarks, mp_hands, mp_drawing, title="", is_original=True, confidence=None):
    # Create a copy of the image
    img = image.copy()
    
    # Different colors for original vs reconstruction
    if is_original:
        landmark_color = (0, 255, 0)  # Green for original
        connection_color = (0, 200, 0)
    else:
        landmark_color = (255, 0, 0)  # Blue for reconstruction
        connection_color = (200, 0, 0)
    
    # Draw the landmarks
    mp_drawing.draw_landmarks(
        img,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=landmark_color, thickness=3, circle_radius=3),
        mp_drawing.DrawingSpec(color=connection_color, thickness=2)
    )
    
    # Add title with background for better visibility
    cv2.rectangle(img, (5, 5), (400, 100), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add confidence score if available
    if confidence is not None:
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(img, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

def process_video(input_path, output_path):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,  # Lower threshold to detect more frames
        min_tracking_confidence=0.3
    )

    # Read the video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer for side-by-side comparison
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Store hand landmarks for PCA
    all_hand_landmarks = []
    frame_data = []  # Store (frame_idx, landmarks, confidence, results) for each frame
    
    print(f"\nFirst pass: collecting hand landmarks...")
    # First pass: collect landmarks for PCA
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Store all hands detected in this frame
            frame_hands = []
            frame_confidences = []
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks_array = []
                for landmark in hand_landmarks.landmark:
                    landmarks_array.extend([landmark.x, landmark.y, landmark.z])
                all_hand_landmarks.append(landmarks_array)
                
                # Get confidence score if available
                confidence = None
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    confidence = results.multi_handedness[idx].classification[0].score
                
                frame_hands.append(hand_landmarks)
                frame_confidences.append(confidence)
            
            frame_data.append((frame_idx, frame_hands, frame_confidences, results))
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            detection_rate = len(frame_data) / frame_idx * 100
            print(f"Processed {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%) - Detection rate: {detection_rate:.1f}%")
    
    if not all_hand_landmarks:
        print("No hand landmarks detected in video")
        return
    
    # Perform PCA with 2 components
    print("\nPerforming PCA analysis...")
    pca = PCA(n_components=2)
    landmarks_array = np.array(all_hand_landmarks)
    pca_result = pca.fit_transform(landmarks_array)
    
    # Reset video capture for second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\nSecond pass: creating comparison video...")
    # Second pass: create comparison video
    landmark_idx = 0  # Index for PCA results
    frame_data_idx = 0  # Index for frame data
    frame_idx = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Create side-by-side comparison
        comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Add frame counter and detection status
        frame_text = f"Frame: {frame_idx}/{total_frames}"
        if frame_data_idx < len(frame_data) and frame_data[frame_data_idx][0] == frame_idx:
            status = f"Tracking {len(frame_data[frame_data_idx][1])} hand(s)"
            color = (0, 255, 0)
        else:
            status = "No Detection"
            color = (0, 0, 255)
        
        status_text = f"{frame_text} - {status}"
        cv2.putText(image, status_text, (width-350, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        comparison[:, :width] = image  # Original on left
        comparison[:, width:] = image  # Will overlay reconstruction on right
        
        # If we have landmarks for this frame
        if frame_data_idx < len(frame_data) and frame_data[frame_data_idx][0] == frame_idx:
            frame_hands = frame_data[frame_data_idx][1]
            frame_confidences = frame_data[frame_data_idx][2]
            
            # Process each hand in the frame
            for hand_idx, (original_landmarks, confidence) in enumerate(zip(frame_hands, frame_confidences)):
                # Reconstruct landmarks using 2 PCA components
                reconstructed_data = pca.inverse_transform(pca_result[landmark_idx:landmark_idx+1])[0]
                
                # Create new landmark proto for reconstruction
                reconstructed_landmarks = landmark_pb2.NormalizedLandmarkList()
                for i in range(21):  # MediaPipe hand has 21 landmarks
                    landmark = landmark_pb2.NormalizedLandmark()
                    landmark.x = reconstructed_data[i*3]
                    landmark.y = reconstructed_data[i*3+1]
                    landmark.z = reconstructed_data[i*3+2]
                    reconstructed_landmarks.landmark.append(landmark)
                
                # Draw original and reconstructed landmarks with different colors
                comparison[:, :width] = draw_hand_landmarks(
                    comparison[:, :width], original_landmarks, mp_hands, mp_drawing,
                    f"Original Hand {hand_idx+1}\n(63 dimensions)", True, confidence)
                comparison[:, width:] = draw_hand_landmarks(
                    comparison[:, width:], reconstructed_landmarks, mp_hands, mp_drawing,
                    f"PCA Reconstruction {hand_idx+1}\n(2 components)", False, confidence)
                
                landmark_idx += 1
            
            frame_data_idx += 1
        
        out.write(comparison)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            detection_rate = frame_data_idx / frame_idx * 100
            print(f"Creating video: frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%) - Detection rate: {detection_rate:.1f}%")
    
    # Release everything
    cap.release()
    out.release()
    print(f"\nComparison video saved as: {output_path}")
    print(f"Overall detection rate: {(len(frame_data) / total_frames * 100):.1f}%")
    
    # Print explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    print("\nExplained variance by first two components:")
    print(f"PC1: {explained_var[0]:.2f}%")
    print(f"PC2: {explained_var[1]:.2f}%")
    print(f"Total: {sum(explained_var):.2f}%")

def main():
    videos = ["IMG_9462.MOV", "IMG_9465.MOV"]
    
    for video in videos:
        print(f"\nProcessing {video}...")
        output_path = f"{os.path.splitext(video)[0]}_comparison.mp4"
        process_video(video, output_path)

if __name__ == "__main__":
    main() 