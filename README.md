# Tunisian Currency Detection with YOLOv11

Real-time detection and recognition of Tunisian Dinar banknotes using YOLOv11 deep learning model.

## Overview

This project uses computer vision to detect and classify Tunisian currency (5, 10, 20, 50 DT banknotes) in real-time through a webcam feed or uploaded images. The system automatically calculates the total value of detected bills.

## Features

- Real-time banknote detection via webcam
- Web interface for image upload
- Automatic total calculation in Tunisian Dinars
- Support for 4 denominations: 5 DT, 10 DT, 20 DT, 50 DT
- High accuracy detection with YOLOv11


## Dataset

- **Total Images**: 2000+
- **Classes**: 4 (5DT, 10DT, 20DT, 50DT)
- **Augmentation**: Rotation, flip, brightness adjustment
- **Split**: 80% train, 10% validation, 10% test

## Installation

```bash
git clone https://github.com/dridi10331/tunisian-currency-detection.git
cd tunisian-currency-detection
pip install -r requirements.txt
```

## Usage

### Webcam Detection
```bash
python yolo11.py
```

### Flask Web Interface
```bash
python app.py
# Open http://localhost:5000
```

## Project Structure

```
├── app.py              # Flask web application
├── yolo11.py           # Webcam detection script
├── best.pt             # Trained YOLOv11 model
├── requirements.txt    # Dependencies
├── templates/
│   └── index.html      # Web interface
└── images/             # Screenshots
```

## Technologies

- Python 3.8+
- YOLOv11 (Ultralytics)
- OpenCV
- Flask
- PyTorch

## How It Works

1. **Detection**: YOLOv11 model identifies banknotes in the frame
2. **Classification**: Each detected bill is classified by denomination
3. **Calculation**: System sums up all detected bills
4. **Display**: Results shown with bounding boxes and total value

## Screenshots

| Detection Example |
|-------------------|
| ![Demo](images/demo.png) |

## Author

Ahmed Omar Dridi

## License

MIT License
