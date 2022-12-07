# Long-term Recurrent Convolutional Network for Emotion Recognition from Photoplethysmography
Attention-LRCN is a deep learning model that recognizes emotional states from photoplethysmography signals. 
* The model extracts temporal features from spectrograms by utilizing a long-term recurrent convolutional network, and a novel attention module is introduced to alleviate the effect of noise components.
* To improve the recognition accuracy, we introduce a weighted knowledge distillation technique, which is a teacher-student learning framework.
![1overall](https://user-images.githubusercontent.com/68531659/203210068-6b9254ca-5819-4361-8ee3-c8ec6cdd5d6e.jpg)


# Experiments
Pytorch code for the Attention-LRCN with WKD technique.
You can download the PPG merged .npy file in [Google Drive](https://drive.google.com/file/d/1u14z3RzUllVeWD5uV4bjE5kv6F-WWZJT/view?usp=sharing). 


To train the Attention-LRCN:
```
python3 baseline_main.py 
```


To train the Attention-LRCN with weighted knowledge distillation:
```
python3 WKD_main.py
```
To inference the results using pre-trained model:
```
python3 inference.py
```
