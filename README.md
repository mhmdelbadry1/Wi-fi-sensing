# Fall Detection

A production-ready setup to train, evaluate, and run a WiFi CSI-based fall detection model, with steps to prepare data and deploy for ESP32-S3 streaming.

- Notebook: /Wifi-Sensing Task/wifi_sensing.ipynb
- Data root: csi-bench/csi-bench-dataset/FallDetection
- Model output: fall_detection_generalizable.pth

## Project Structure
- venv/                  Virtual environment (recommended)
- requirements.txt       Python dependencies
- csi-bench/             Dataset download and extracted contents
  - csi-bench-dataset/FallDetection
- checkpoints/           Stage checkpoints during training
- fall_detection_generalizable.pth  Best model checkpoint
- Wifi-Sensing Task/     Jupyter notebook and experiments

## Prerequisites
- Python 3.10+ (3.12 supported)
- CUDA-capable GPU (optional but recommended)
- pip or conda

## Setup
1. Create and activate virtualenv
   - Linux/macOS:
     - python -m venv venv
     - source venv/bin/activate
   - Windows PowerShell:
     - python -m venv venv
     - .\venv\Scripts\Activate.ps1

2. Install dependencies
   - pip install -r requirements.txt

3. Verify environment
   - python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"

## Downloading the CSI-Bench dataset
The notebook includes a temporary signed URL that expires. If it fails (403/404), use one of the options below.

- Option A: Temporary signed link (quick, may expire)
  - wget -O csi-bench.zip "PASTE_THE_SIGNED_URL_HERE"
  - mkdir -p csi-bench/FallDetection/csi-bench-dataset
  - unzip csi-bench.zip "*/FallDetection/*" -d csi-bench

- Option B: Kaggle API (recommended)
  - pip install kaggle
  - Place kaggle.json in ~/.kaggle/ (Linux/macOS) or %USERPROFILE%\.kaggle\ (Windows)
  - kaggle datasets download -d <dataset-owner>/<dataset-name>
  - unzip the archive and move the FallDetection folder into csi-bench/csi-bench-dataset/

- Option C: Manual download
  - Download the archive via browser and extract FallDetection into csi-bench/csi-bench-dataset/

After extraction, you should have:
- csi-bench/csi-bench-dataset/FallDetection/metadata/sample_metadata.csv
- csi-bench/csi-bench-dataset/FallDetection/splits/*.json
- csi-bench/csi-bench-dataset/FallDetection/**/*.h5

## Run from Notebook (exploration and training)
- Open: /Wifi-Sensing Task/wifi_sensing.ipynb
- The notebook covers:
  - Data inspection and visualizations
  - Preprocessing to standardized (232, 500) CSI arrays
  - ViT-based model with domain adaptation and curriculum learning
  - Training, validation, and evaluation on Easy/Medium/Hard splits
  - Saving best model to fall_detection_generalizable.pth

## CLI-style Usage (optional)
If you prefer running training or inference from scripts, adapt the notebook cells into Python scripts (train.py, infer.py). Ensure CONFIG paths match:
- TASK_PATH: csi-bench/csi-bench-dataset/FallDetection
- MODEL_SAVE_PATH: fall_detection_generalizable.pth
- CHECKPOINT_DIR: checkpoints

### Training (high-level)
- Ensure dataset is extracted as above
- Run the curriculum training section in the notebook
- Best model checkpoint is saved automatically when validation improves

### Evaluation
- Notebook provides evaluation on Easy/Medium/Hard test sets and saves confusion matrices (PNG files).

### Inference
- Load fall_detection_generalizable.pth and run the inference section in the notebook using standardized CSI input shape (232, 500).

## ESP32-S3 Real-time Notes
- The model runs on a host machine (Python/PyTorch). ESP32-S3 streams CSI packets to the host.
- Recommended pipeline:
  - ESP32-S3 firmware: capture CSI and send over WiFi/Serial
  - Host receiver: parse CSI into (subcarriers x time) window
  - Preprocess to target shape (232, 500) and normalize (device-aware optional)
  - Forward through GeneralizableFallDetector for prediction and uncertainty
- Ensure latency targets by batching small windows (e.g., 500 time samples) and using torch.no_grad().

## Tips
- If DataLoader errors occur in notebooks, set num_workers=0.
- For GPU training stability, reduce BATCH_SIZE if you hit OOM.
- Hard set accuracy is the deployment indicator; aim ≥70%.

## Troubleshooting
- Signed URL expired: use Kaggle API or manual download.
- File not found in metadata: the notebook auto-resolves via basename search.
- Multiprocessing errors in notebooks: use num_workers=0, pin_memory=False for evaluation.

## License
- Ensure compliance with dataset license. This project code is provided as-is.