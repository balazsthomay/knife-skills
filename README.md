# Knife Safety Monitor

## What It Does

Computer vision system that monitors knife safety in real-time using:
- **Hand tracking** (21 landmarks per hand)
- **Knife detection** with precise blade segmentation
- **3D depth awareness** for accurate distance measurements
- **Real-time safety alerts** with severity levels

## Key Features

### Advanced Hand Detection
- MediaPipe Hands integration
- Dual hand tracking with handedness detection
- 21 landmarks per hand

### Custom Knife Detection
- fine-tuned YOLOv11 segmentation model
- Precise blade boundary extraction
- Tip and handle point identification

### 3D Depth-Aware Safety
- Core ML Depth Anything V2 optimized for Apple Silicon
- 3D distance calculations
- Filters false positives from 2D overlap

## Demo

![Knife Safety Monitor Demo](demo.gif)

*Real-time knife safety monitoring with hand tracking, blade detection, and 3D depth-aware safety alerts*

## Quick Start

```bash
# Run the trainer
python main.py
```

The application will open a real-time video feed with overlays showing:
- Hand landmarks and tracking
- Knife detection with blade boundaries
- Safety warnings with distance measurements
- 3D depth visualization (optional)

## Contributing

This is a proof-of-concept demonstrating advanced computer vision techniques. Areas for enhancement:

- Bigger model for knife segmentation, additional knife types and orientations
- Multi-camera support for better depth accuracy