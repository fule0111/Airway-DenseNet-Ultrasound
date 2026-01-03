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
Recommended environment:
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy, Matplotlib, scikit-learn  
- Graphviz (optional for network visualization)

To install dependencies:
### 方式1：推荐（用我们刚创建的 requirements.txt 一键安装）
```bash
pip install -r requirements.txt
```

### 方式2：手动安装（备用，如需单独安装）
```bash
pip install torch==1.10.2 torchvision==0.11.3 numpy==1.21.6 matplotlib==3.4.3 scikit-learn==1.0.2
```
---
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
**Note:** This script adopts the core fusion strategy used in the paper (consistent with the paper's experimental design).  
**Input:** 预测输出文件需满足统一格式（每个子模型输出为 `{model_name}_predictions.csv`，包含3列：`sample_id`（样本编号）、`pred_prob`（预测概率）、`true_label`（真实标签））  
**Supported Fusion Strategy (Core Strategy in the Paper):**  
采用「5折交叉验证+4个平面准确率比值加权融合」
- 5折交叉验证：基于5折交叉验证的验证集结果，统计各模型的稳定准确率
- 加权融合：以4个超声平面模型（MPM/TPH/TPT/PSPL）的验证集准确率为比值，对各模型的预测结果进行加权赋值，最终生成融合结果
**Usage:**
```bash
python ensemble.py ./models/  # 直接运行即可
```
---

### 4.3 train.csv / test.csv
These two files are the core log files of the project (consistent with the 5-fold cross validation in the paper):
- `train.csv`: Records the training process details and validation results of the four sub-models (MPM/TPH/TPT/PSPL), including key indicators such as epoch, loss and accuracy.
- `test.csv`: Records the final testing results of the four sub-models and the weighted fusion model, which can be used to verify the experimental data in the paper.
- Format explanation: Both files are in CSV format, which can be directly opened and viewed with Excel, WPS or Notepad (no additional software required).
---

## 5. Pre-trained Model Weights Availability
### 5.1 Why This Part Is Important
The pre-trained weights of the four sub-models (MPM/TPH/TPT/PSPL) are necessary to run the `ensemble.py` script (without these files, the fusion function cannot be executed normally).

### 5.2 Access Method
Due to the protection of research results and file size limitations, the pre-trained weights are not stored in this GitHub repository. 
**Researchers who need the weights for academic research can apply by contacting the corresponding author:**
- E-mail: Fuchunmeng918@163.com
- Application Note: Please specify the research purpose and institution in the email subject.

### 5.3 Usage Note After Obtaining Weights
1. Create a folder named `models` in the project root directory (same folder as `README.md` and `ensemble.py`);
2. Place the obtained weight files into `./models/`, and ensure the file names are consistent with the following format:
   - MPM model: `MPM_pretrained.pth`
   - TPH model: `TPH_pretrained.pth`
   - TPT model: `TPT_pretrained.pth`
   - PSPL model: `PSPL_pretrained.pth`

---
## 6. Model Concept
The project adopts a **multi-plane fusion strategy**, including the following four sub-models:
- **MPM:** Mid-Plane of the Mandible (sagittal)  
- **TPH:** Transverse Plane at the Hyoid Bone  
- **TPT:** Transverse Plane at the Thyroid Cartilage  
- **PSPL:** Para-Sagittal Plane of the Larynx  
The `ensemble.py` script merges predictions from these models into a unified model to enhance classification and airway visualization performance.

---
## 7. Citation
If you use this project in your research, please cite:  
**"A New Deep Learning-Driven Strategy for Airway Management: The 'Two-Models-Three-Steps' Decision Framework."**

---
## 8. License & Permanent Access
### 8.1 Open Source License
This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file in the project root directory for details. This license allows free use, modification and distribution for academic research purposes.

### 8.2 Permanent Project Access
To ensure the long-term preservation and accessible of the project (in line with journal requirements), this project will be archived on Zenodo to obtain a persistent Digital Object Identifier (DOI):
- Persistent DOI (to be filled after archiving): (https://doi.org/10.5281/zenodo.18140439)
