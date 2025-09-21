# Parking Management Video Processing

This project performs automated parking space detection and management on a parking lot video using a custom YOLOv8s model trained on the **VisDrone** dataset and predefined parking zones. The output video highlights parked vehicles and available spaces in real-time.


---

## Overview

The application processes a video of a parking lot to detect the occupancy status of parking spaces. It leverages the `ultralytics` package's `ParkingManagement` solution combined with a YOLOv8s model fine-tuned on the VisDrone dataset for accurate vehicle detection. The parking zones are defined via a JSON annotations file.

---

## Features

- Real-time vehicle detection in designated parking zones.
- Custom YOLOv8s model (`best.pt`) trained on the VisDrone dataset for robust vehicle detection.
- Configurable parking zones via JSON annotations (`parking_zones.json`).
- Outputs an annotated video showing parking occupancy status.
- Supports common video input and output formats.

---

## Prerequisites

- Python 3.7 or higher
- OpenCV (`cv2`)
- Ultralytics package (`ultralytics`)
- Trained YOLOv8s model file (`best.pt`), trained on VisDrone dataset
- Parking zones annotations JSON file (`parking_zones.json`)

---

## Installation

1. Clone or download this repository.

2. Install required dependencies:

```bash
pip install opencv-python ultralytics
