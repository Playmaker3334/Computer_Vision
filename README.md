# Computer Vision Projects Repository

Welcome to the **Computer Vision Projects** repo! This collection brings together several end-to-end demos and experiments using deep learning and classical CV techniques.

## 📂 Repository Structure

```
.
├── Brain_Tumor_Detection   # CNN-based MRI brain tumor classifier  
├── CNN_Facial_emotions     # Classify facial expressions with a convolutional network  
├── Clothes_CNN             # Clothing item recognition on fashion images  
├── Face_Detection          # Classic HAAR / DNN face detection examples  
├── Yolo_human_detection    # Real-time person detection with YOLO  
└── LICENSE                 # GPL-3.0 open source license
```

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Playmaker3334/Computer_Vision.git
   cd Computer_Vision
   ```

2. **Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate.bat   # Windows
   ```

3. **Install dependencies**  
   Each project folder contains its own \`requirements.txt\` or notebook-specific imports. For example, to install for the Brain Tumor Detector:
   ```bash
   pip install -r Brain_Tumor_Detection/requirements.txt
   ```

4. **Run a demo**  
   - **Brain Tumor Detection**:  
     ```bash
     cd Brain_Tumor_Detection
     python detect_tumor.py --model model.pth --input scans/
     ```
   - **Facial Emotions** (Jupyter notebook):  
     ```bash
     cd CNN_Facial_emotions
     jupyter notebook Facial_Emotions_CNN.ipynb
     ```
   - **YOLO Human Detection**:  
     ```bash
     cd Yolo_human_detection
     python yolo_detect.py --weights yolov5s.pt --source video.mp4
     ```

## 🔍 Highlights

- **Deep Learning**: Convolutional neural nets for image classification & detection.  
- **Real-Time Inference**: YOLO-based human detection.  
- **Classic CV**: Haar cascades and OpenCV pipelines.  
- **Jupyter Notebooks**: Interactive exploration and training.  

## 📜 License

This project is licensed under the **GPL-3.0** License. See the [LICENSE](LICENSE) file for details.
