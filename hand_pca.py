import cv2
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def extract_hand_features(input_path):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Read the video
    cap = cv2.VideoCapture(input_path)
    
    # Store hand landmarks for each frame
    all_hand_landmarks = []
    frame_count = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Extract hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to flat array of coordinates
                landmarks_array = []
                for landmark in hand_landmarks.landmark:
                    landmarks_array.extend([landmark.x, landmark.y, landmark.z])
                all_hand_landmarks.append(landmarks_array)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    return np.array(all_hand_landmarks)

def perform_pca_analysis(features, n_components=10):
    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform the data
    pca_result = pca.fit_transform(features)
    
    return pca, pca_result

def plot_pca_results(pca, pca_result, video_name):
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    explained_var = pca.explained_variance_ratio_ * 100
    plt.bar(range(1, len(explained_var) + 1), explained_var)
    plt.title(f'Explained Variance Ratio\n{video_name}')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    cumulative_var = np.cumsum(explained_var)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
    plt.title(f'Cumulative Explained Variance\n{video_name}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    
    # Save the plot
    plot_filename = f"pca_analysis_{os.path.splitext(video_name)[0]}.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"PCA analysis plot saved as: {plot_filename}")
    
    # Print explained variance for first few components
    print("\nExplained variance by component:")
    for i, var in enumerate(explained_var[:5], 1):
        print(f"PC{i}: {var:.2f}%")

def main():
    videos = ["IMG_9462.MOV", "IMG_9465.MOV"]
    
    for video in videos:
        print(f"\nProcessing {video}...")
        
        # Extract features
        features = extract_hand_features(video)
        
        if len(features) == 0:
            print(f"No hand landmarks detected in {video}")
            continue
            
        print(f"Collected {len(features)} hand landmark samples")
        
        # Perform PCA
        pca, pca_result = perform_pca_analysis(features)
        
        # Plot and save results
        plot_pca_results(pca, pca_result, video)
        
        # Save PCA components and transformed data
        np.save(f"pca_components_{os.path.splitext(video)[0]}.npy", pca.components_)
        np.save(f"pca_transformed_{os.path.splitext(video)[0]}.npy", pca_result)

if __name__ == "__main__":
    main() 