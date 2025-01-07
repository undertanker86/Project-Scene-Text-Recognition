# OCR Project: Scene Text Recognition

## Introduction

This project focuses on **Scene Text Recognition**, which involves detecting and recognizing text within images captured from real-world environments. It has various practical applications, including:

- Text processing in images (e.g., documents, signage).
- Data collection by extracting text from online images.
- Automating processes such as order handling and payment processing.

The solution consists of two main components:

1. **Text Detection (YOLOv11)**: Identifies the locations of text blocks in images.
2. **Text Recognition (CRNN)**: Decodes the text content within identified regions.

## Project Workflow

1. **Input**: An image containing text.
2. **Output**: Coordinates and text content of the detected regions.

## Dataset

The project utilizes the **ICDAR2003 dataset** for training and testing. Preprocessing is required to convert the data into formats compatible with YOLOv11 and CRNN models.

### Dataset Preparation

1. Extract data from the XML format to retrieve image paths, bounding boxes, and labels.
2. Convert bounding box coordinates to YOLO format.
3. Organize the data into the YOLOv11 directory structure:
   - `images/`: Contains input images.
   - `labels/`: Contains corresponding text labels.
4. Split data into training, validation, and test sets (70:20:10 ratio).

## Model Architecture

### Text Detection
- **YOLOv11**: A single-stage object detection model optimized for identifying text locations.

### Text Recognition
- **CRNN (Convolutional Recurrent Neural Network)**:
  - **CNN**: Extracts image features.
  - **RNN**: Processes sequential data for character prediction.
  - **CTC Loss**: Handles alignment between predictions and ground truth.

## Implementation Steps

### Text Detection Module

1. Install required libraries, including the YOLOv11 framework (`ultralytics`).
2. Train the YOLOv11 model on the processed dataset using the `ultralytics` library.
3. Evaluate the model for accuracy and performance.

### Text Recognition Module

1. Prepare OCR-specific datasets by cropping text regions and associating them with labels.
2. Train the CRNN model using PyTorch with pre-trained ResNet and Bi-LSTM layers.
3. Evaluate the model using validation and test sets.

### Full Pipeline Integration

1. Use YOLOv11 to detect text regions.
2. Crop detected regions and pass them to the CRNN model for recognition.
3. Combine results to provide final outputs.

## Results

### Performance Metrics

- **Detection**: Evaluate precision and recall for bounding box predictions.
- **Recognition**: Evaluate character-level and word-level accuracy.

### Example Outputs

Annotated images showcasing bounding boxes and recognized text.
![result](https://i.imgur.com/34nnH3F.png)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/undertanker86/Project-Scene-Text-Recognition.git
   ```
2. Prepare the dataset:
   - Download the ICDAR2003 dataset.
   - Follow the preprocessing steps outlined above.

3. Train the models:
   ```bash
   python detect_yolov11.py  # For YOLOv11
   python recognition_crnn.py  # For CRNN
   ```

## Usage

1. Run the pipeline:
   ```bash
   python run_pipeline.py --image <image-path>
   ```

2. Visualize results:
   The program will display the input image with annotated text regions and their recognized content.



## References

- [ICDAR2003 Dataset](https://drive.google.com/file/d/1kUy2tuH-kKBlFCNA0a9sqD2TG4uyvBnV/view)
- [YOLOv11 Documentation](https://github.com/ultralytics/ultralytics)
- [CRNN Research Paper](https://arxiv.org/abs/1507.05717)

---
For more details, refer to the project report or contact the contributors.

