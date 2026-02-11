# TERDNet

This repository represents the official implementation of the paper titled "TERDNet: Transformer Encoder-Recurrent Decoder Network for Scene Change Detection (ICRA 2026)".

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/AutoCompSysLab/TERDNet/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.12-red)](https://pytorch.org/)
[![ICRA 2026](https://img.shields.io/badge/ICRA-2026-brightgreen)](https://github.com/AutoCompSysLab/TERDNet)


## 📌 Overview

![TERDNet Pipeline](assets/pipeline_overview.png)


## 🔧 Installation

### Requirements

- Python ≥ 3.8  
- PyTorch  
- numpy, scipy  
- open3d  
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
````

## 📦 Dataset Preparation

This repository does **not include dataset files**.
Please prepare datasets manually and set the root path in:

```
src/dataset/path_config.py
```

Example structure:

```
datasets/
    VL_CMU_CD/
        train/
        val/
        test/
        annotations.json
```

## 🚀 Usage

### 📍 Train

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

## 📊 Outputs & Logs

Once training runs, outputs are stored under:

```
outputs/
    run_<timestamp>/
        checkpoints/
        logs/
        metrics.json
```

## 📸 Example Visualizations

### Qualitative Predictions

![Qualitative Results](assets/qualitative_examples.png)

Qualitative examples showing TERDNet predictions compared with ground truth.

### Dataset Visualization

![Dataset Example](assets/dataset_example.png)

Input point clouds and annotations for the benchmark dataset.

## 🧠 Acknowledgements

We thank the developers of **Segment Anything (SAM)**, which provided foundation model support for segmentation tasks used in our pipeline, and we acknowledge that parts of this code were adapted and modified from **C-3PO** and related implementations to build the core of TERDNet.
This work benefited from prior open-source contributions in dynamic perception and scene change detection research.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yourlastname2026terdnet,
  title={TERDNet: Transformer-based Dynamic Perception Network},
  author={Your Name and Coauthors},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

## 🪄 Acknowledgement

✅ We sincerely acknowledge 
✅ We also thank Segment Anything for providing an excellent vision foundation model.

