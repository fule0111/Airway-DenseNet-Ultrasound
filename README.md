# AirwayDisNetPublic

## 1. Project Overview
**AirwayDisNetPublic** is a deep learning–based research framework for airway management. 
It aims to achieve intelligent recognition and decision support for difficult airways through the integration of multi-plane ultrasound imaging models.  
The core concept introduces a new **“Two-Models-Three-Steps”** deep learning–driven strategy to combine features from multiple ultrasound planes for improved prediction accuracy.

---

## 2. Project Structure
```
AirwayDisNetPublic/
│
├── plot.py             # Plot training and testing loss/error curves
├── ensemble.py         # Model fusion script (ensemble different plane models)
├── train.csv / test.csv# Training and testing log files
├── README.md           # Project description (this file)
```

---

## 3. Environment Requirements
To install dependencies:
# 方式1：推荐（用我们刚创建的 requirements.txt 一键安装）
pip install -r requirements.txt

# 方式2：手动安装（备用，如需单独安装）
pip install torch==1.10.2 torchvision==0.11.3 numpy==1.21.6 matplotlib==3.4.3 scikit-learn==1.0.2


## 4. Main Files Description

### 4.1 plot.py  
Plots training and validation loss/error curves.  
**Usage:**
```bash
python plot.py ./experiment_folder/
```
**Output:**
- `loss.png` — Cross-entropy loss curve  
- `error.png` — Classification error curve  

---

### 4.2 ensemble.py  
Performs model ensemble across different ultrasound plane models.  
**Input:** prediction outputs from each sub-model (e.g., MPM, TPH, TPT, PSPL).  
**Output:** fused results using mean or voting strategy.

**Usage:**
```bash
python ensemble.py ./models/
```

---

## 5. Model Concept
The project adopts a **multi-plane fusion strategy**, including the following four sub-models:
- **MPM:** Mid-Plane of the Mandible (sagittal)  
- **TPH:** Transverse Plane at the Hyoid Bone  
- **TPT:** Transverse Plane at the Thyroid Cartilage  
- **PSPL:** Para-Sagittal Plane of the Larynx  

The `ensemble.py` script merges predictions from these models into a unified model to enhance classification and airway visualization performance.

---

## 6. Citation
If you use this project in your research, please cite:  
**"A New Deep Learning-Driven Strategy for Airway Management: The 'Two-Models-Three-Steps' Decision Framework."**
