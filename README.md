# Long-term Recurrent Convolutional Network for Emotion Recognition from Photoplethysmography

Pytorch implementation of Long-term Recurrent Convolutional Network for Emotion Recognition from Photoplethysmography.

![1overall](https://user-images.githubusercontent.com/68531659/203210068-6b9254ca-5819-4361-8ee3-c8ec6cdd5d6e.jpg)

## Overview
Emotion recognition is important for regulating stress levels and maintaining mental health, and it is known that emotional states can be inferred from physiological signals.
However, in practice, the problem of emotion recognition contains many challenges due to various types of external noise and different individual characteristics.
This paper proposes a deep learning model called Attention-LRCN for recognizing emotional states from photoplethysmography signals.
The proposed model extracts temporal features from spectrograms by utilizing a long-term recurrent convolutional network, and a novel attention module is introduced to alleviate the effect of noise components.
Moreover, to improve the recognition accuracy, we propose a weighted knowledge distillation technique, which is a teacher-student learning framework.
We quantify the uncertainty of teacher’s predictions, and the predictive uncertainty is utilized to adaptively compute the weight of the distillation loss.
To demonstrate the effectiveness of the proposed method, experiments were conducted on the WESAD dataset, which is a public dataset for stress and affect detection.
We also collected our own dataset from 34 subjects to verify the accuracy of the proposed method.
Experimental results demonstrate that the proposed method significantly outperforms previous algorithms on both the public and real-world datasets.

## Installation
#### Requirements
- Ubuntu 18.04
- Python v3.7.11
- Pytorch v1.7.1

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
python3 Attention-LRCN.py 
```

To train the Attention-LRCN with weighted knowledge distillation:
```
python3 Weighted KD.py
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
python3 merge_PPG.py
``` 
to get merged PPG data as input to the Attention-LRCN model.
