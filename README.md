# TERDNet

![TERDNet Pipeline](assets/pipeline_overview.png)

**Official implementation of TERDNet**  
Transformer-based Dynamic Perception Network  
(*Accepted at IEEE International Conference on Robotics and Automation — ICRA 2026*)


![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.12-red)

---

## 🚀 Overview

TERDNet is a transformer-based dynamic perception framework for **scene change detection** and **per-point classification** in 3D point clouds.  
This repository contains the **official code** used in the ICRA 2026 paper, including dataset interfaces, model definitions, training scripts, and evaluation tools.

---

## ✨ Key Features

- Transformer backbone for 3D representation learning  
- Multi-scale feature fusion with pyramid levels  
- Iterative decoding for structured predictions  
- Dataset integration and evaluation utilities

---

## 🛠 Installation

### Requirements

- Python ≥ 3.8  
- PyTorch  
- numpy, scipy, open3d, tqdm

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 📦 Dataset Preparation

This repository does **not include dataset files**.
Please download and prepare datasets manually. Update your dataset root in:

```
src/dataset/path_config.py
```

Example dataset layout:

```
datasets/
    VL_CMU_CD/
        train/
        val/
        test/
        annotations.json
```

---

## 🎯 Usage

### 🧠 Train

```bash
python -u src/train.py \
    --train-dataset VL_CMU_CD \
    --test-dataset VL_CMU_CD \
    --data-cv 0 \
    --input-size 1024 \
    --model vit_b_terdnet \
    --fusion-type ours \
    --pyramid-levels 4 \
    --decoder-iters 3 \
    --warmup \
    --loss-weight
```

### Supported Options

| Flag               | Description                  |
| ------------------ | ---------------------------- |
| `--train-dataset`  | Dataset name for training    |
| `--test-dataset`   | Dataset name for evaluation  |
| `--data-cv`        | Cross-validation split index |
| `--input-size`     | Number of input points       |
| `--model`          | Model backbone               |
| `--fusion-type`    | Feature fusion method        |
| `--pyramid-levels` | Pyramid feature levels       |
| `--decoder-iters`  | Iterative decoding steps     |
| `--warmup`         | Warmup loss scheduling       |
| `--loss-weight`    | Weighted loss                |

---

## 📊 Outputs & Logs

Training outputs including checkpoints, logs, and evaluation metrics are saved here:

```
outputs/
    run_<timestamp>/
        checkpoints/
        logs/
        metrics.json
```

---

## 📸 Example Visualizations

### Qualitative Results

![Qualitative Results](assets/qualitative_examples.png)

Sample network predictions vs ground-truth labels on benchmark datasets.

### Dataset Example

![Dataset Example](assets/dataset_example.png)

Visualization of input point clouds and annotations.

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourlastname2026terdnet,
  title={TERDNet: Transformer-based Dynamic Perception Network},
  author={Your Name and Coauthors},
  booktitle={Proc. of IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

---

## 📄 License

This implementation is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 💬 Contributing

Contributions, issues, and feature requests are welcome!
Please open issues or pull requests.

---

## 📬 Contact

For questions or support, contact:
[your.email@institution.edu](mailto:your.email@institution.edu)

````

---

## 🧠 Quick Notes Before Publishing

### 📌 Add image assets

Create an `assets/` folder and add images:

| File name                     | Purpose                          |
|------------------------------|----------------------------------|
| `pipeline_overview.png`      | Model architecture illustration  |
| `qualitative_examples.png`   | Example predictions vs GT        |
| `dataset_example.png`        | Dataset visualization            |

Then the images will render automatically.

---

### 📌 Replace placeholders

Before the public release, update:

✔ `yourlastname2026terdnet` → your actual BibTeX citation key  
✔ `Your Name and Coauthors` → full author list  
✔ `<your.email@institution.edu>` → your contact email

---
