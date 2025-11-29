# Driver Drowsiness Detection

Real-time driver drowsiness detection using MediaPipe face mesh + Eye Aspect Ratio (EAR). When the driver's eyes remain closed for several consecutive frames, the system raises an audible alarm and logs the event.

## Features
- Real-time webcam detection
- Eye Aspect Ratio (EAR) measure using MediaPipe
- Configurable thresholds
- Optional alarm WAV file or fallback beep

## Requirements
Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
