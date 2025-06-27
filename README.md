# Hand Motion PCA Analysis

This project analyzes hand motion data using Principal Component Analysis (PCA) to understand and reconstruct hand movements from video.

## Overview

The system:
1. Detects hand landmarks using MediaPipe
2. Applies PCA to reduce 63 dimensions (21 landmarks × 3 coordinates) to 10 principal components
3. Reconstructs hand movements and visualizes the results
4. Provides detailed analysis of reconstruction quality

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Videos**:
   - Place training videos with suffix `-train.MOV`
   - Place test video with suffix `-test.MOV`

2. **Run Analysis**:
```bash
python hand_pca_evaluate.py
```

3. **Outputs**:
   - `test_comparison.mp4`: Side-by-side comparison of original vs reconstructed hand movements
   - `hand_pca_metrics.png`: Plots showing error distribution and explained variance
   - `hand_pca_model.pkl`: Saved PCA model (created after first run)
   - `hand_scaler.pkl`: Saved data scaler (created after first run)

## Results

The PCA model:
- Uses 10 principal components
- Captures ~99.42% of variance in hand movements
- Achieves low reconstruction error on test data

## Technical Details

### Hand Detection
- Uses MediaPipe Hands for landmark detection
- Tracks 21 3D landmarks per hand
- Handles video rotation and frame synchronization

### PCA Implementation
- Standardizes coordinates before PCA
- Reduces from 63D to 10D
- Provides frame-by-frame reconstruction quality metrics

### Visualization
- Side-by-side comparison of original and reconstructed hands
- Real-time error metrics
- Clear status indicators for detection quality

## File Structure

```
.
├── README.md
├── requirements.txt
├── hand_pca_evaluate.py    # Main analysis script
├── models/                 # Saved models
│   ├── hand_pca_model.pkl
│   └── hand_scaler.pkl
├── data/                   # Video data
│   ├── train/
│   └── test/
└── outputs/               # Generated outputs
    ├── videos/
    └── metrics/
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License # CMDetection
