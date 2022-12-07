# Long-term Recurrent Convolutional Network for Emotion Recognition from Photoplethysmography

Pytorch implementation of Long-term Recurrent Convolutional Network for Emotion Recognition from Photoplethysmography.

![1overall](https://user-images.githubusercontent.com/68531659/203210068-6b9254ca-5819-4361-8ee3-c8ec6cdd5d6e.jpg)

## Overview

## Method

## Installation
#### Requirements
- 
#### Environment Setup
- We recommend to conda for installation. Create the conda environment by running:
```
conda create -n attentionLRCN python=3.8
conda activate attentionLRCN
pip install -r requirements.txt
```

#### Running Code
To train the Attention-LRCN:
```
python3 baseline_main.py 
```

To train the Attention-LRCN with weighted knowledge distillation:
```
python3 WKD_main.py
```

We provide pretrained weights to reproduce results in the paper. You can download the pretrained weights in models.
To inference the emotion recognition results using our pre-trained model:
```
python3 inference.py
```
#### Dataset
We use publically available WESAD dataset for evaluation. You can download the dataset [here](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29).
Put the dataset in the directory with the file name "data/", and run:
```
python3 merged_PPG.py
``` 
to get merged PPG data.
