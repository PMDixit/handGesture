# Hand Gesture Recognition System

A real-time hand gesture recognition system that can classify American Sign Language (ASL) and Indian Sign Language (ISL) gestures using computer vision and deep learning.

## Features

- **Real-time Recognition**: Live camera feed with instant gesture classification
- **Multi-Dataset Support**: 
  - American Sign Language (ASL) - 28 classes
  - Indian Sign Language (ISL) - 36 classes
- **Advanced Preprocessing**: 
  - K-means clustering for image segmentation
  - Morphological operations (erosion, dilation)
  - Adaptive thresholding
  - Gaussian blur and noise reduction
- **Deep Learning Model**: MobileNet V2 with quantization for efficient inference
- **GUI Interface**: PyQt5-based user interface for easy interaction
- **Configurable Parameters**: Adjustable preprocessing parameters in real-time

## Project Structure

```
handGesture/
├── datasets/
│   ├── american/          # ASL dataset (A-Z, Space, Nothing)
│   └── indian/           # ISL dataset (0-9, A-Z)
├── models/               # Trained model weights
├── training/             # Training notebooks and scripts
│   ├── IndianFinal.ipynb
│   ├── IndianResnet970.ipynb
│   ├── MobilenetIndian70img.ipynb
│   └── resnet9with400x400.ipynb
├── handDetect.py         # Main detection and classification logic
├── model.py             # Model utilities and prediction functions
├── Ui.py                # PyQt5 GUI interface
└── play.py              # Additional utilities
```

## Requirements

- Python 3.7+
- OpenCV
- PyTorch
- PyQt5
- cvzone (HandTrackingModule)
- NumPy
- torchvision

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd handGesture
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models to the `models/` directory:
   - `MobileNet_V2Indian70img.pth` - Indian Sign Language model
   - `MobileNet_V2ASLNotEroded.pth` - American Sign Language model

## Usage

### Running the Application

```bash
python Ui.py
```

### Key Features

1. **Model Selection**: Choose between Indian (ISL) or American (ASL) sign language
2. **Real-time Detection**: Place your hand in the detection box for instant recognition
3. **Sentence Formation**: The system accumulates recognized gestures to form complete words/sentences
4. **Parameter Adjustment**: Modify preprocessing parameters in real-time:
   - Erosion iterations
   - Dilation iterations
   - K-means clustering value
   - Manual box positioning

### Detection Box Controls

- **Automatic**: Hand tracking with MediaPipe
- **Manual**: Use arrow keys to adjust box position
- **Size**: Fixed 250x250 pixel detection area

## Technical Details

### Preprocessing Pipeline

1. **Image Capture**: Real-time camera feed
2. **ROI Extraction**: Crop hand region from detection box
3. **K-means Clustering**: Optional segmentation (k < 101)
4. **Grayscale Conversion**: Convert to single channel
5. **Gaussian Blur**: Noise reduction
6. **Adaptive Thresholding**: Binary image creation
7. **Morphological Operations**: Opening, closing, dilation, erosion
8. **Resize**: Scale to 128x128 for model input

### Model Architecture

- **Base Model**: MobileNet V2 (pretrained=False)
- **Classifier**: Custom final layer for target classes
- **Optimization**: Dynamic quantization (INT8)
- **Inference**: TorchScript for production deployment

### Dataset Classes

#### American Sign Language (28 classes)
- Letters: A-Z
- Special: Space, Nothing

#### Indian Sign Language (36 classes)
- Numbers: 0-9
- Letters: A-Z

## Acknowledgments

- OpenCV for computer vision capabilities
- PyTorch for deep learning framework
- MediaPipe for hand tracking
- cvzone for simplified hand detection implementation

## Contact

For questions and support, please open an issue on the GitHub repository. 