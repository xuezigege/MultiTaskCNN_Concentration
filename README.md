# Deep Learning–Enhanced DNAzyme-Driven Rolling-Circle Amplification Encoding for Multibacterial Detection

---

## Overview

This repository contains the **complete deep learning pipeline** used for simultaneous detection of multiple bacterial species via images.  
The model is a multi‑task convolutional neural network (CNN) that:

- Classifies the presence/absence of three bacterial targets  
- Predicts the continuous concentrations of each target (regression)  

The code includes data loading, model definition, training, evaluation, and inference on new images.  
A pre‑trained best model (`best_concentration_model.pth`) is provided.

---

## Repository Structure
```
.
├── multitask_cnn.py # Full training and evaluation script
├── best_concentration_model.pth # Pre‑trained model weights (best on validation set)
├── requirements.txt # Python dependencies
└── README.md
```

## Dataset Information

The full dataset used in the paper is not publicly available due to ongoing related studies. However, it can be obtained from the corresponding author upon reasonable request for academic research purposes.
If you wish to train or test the model on your own data, please prepare your dataset in the following format.

### Data Format
The training/validation/test sets should be organized as separate folders (e.g., `train/`, `val/`, `test/`), each containing:
- A CSV file with the same name as the folder (e.g., `train.csv`), containing the following columns:
  - `image_path`: absolute or relative path to the image file
  - `x`, `y`, `z`: continuous concentration values (used for both classification and regression)
- The actual image files (e.g., `.jpg`, `.png`) referenced in the CSV.
**Important:** The code automatically converts continuous concentrations into binary classification labels. Ensure your data follows this logic.

## How to Use
1. Clone or download this repository. 

2. Install the required Python packages (requirements.txt). 

3. Use `multitask_cnn.py` to train the model on your own data (modify `base_dir` as needed). 

4. Load `best_concentration_model.pth` and use `predict_unknown_image` for inference on new images.  


## License
All code  is released for academic and non-commercial use.

## Contact
For questions about the code or data availability, please contact: lijiuxing@dlut.edu.cn; yychang@dlut.edu.cn; mliu@dlut.edu.cn.
