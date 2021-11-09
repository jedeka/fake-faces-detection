# Fake Faces Detection -- 20hrs Challenge
Two approaches for detecting fake faces (_e.g._ DeepFake) using EfficientNet and Vision Transformers. Due to local resource restrictions, Google Colab is utilized (with GPU support). 

Note that this is a timed challenge, thus the settings are not the most optimal (_e.g._ without hyperparameter tuning, data augmentation, etc.), and we ignore time features as the project is doing more image classification task rather than object detection and tracking.  

### Dataset
* ![Click here for the split dataset](https://drive.google.com/drive/folders/1RrDFPuDWJtM-D8Tri_crTPpOmf_n0nT3?usp=sharing)  
(Split by sklearn's `train_test_split` with train:test = 0.8 : 0.2. Using this data is recommended, most of the approach I wrote are using this data, to save time)  
* ![Click here for the full image dataset](https://drive.google.com/drive/folders/1TyjYmiyRoo7WQoqIX2X0P5zPfjTdUi5j?usp=sharing)


### Setup
* Google Colab  
To run this code at Google Colab, clone this repo and drag to Google Drive, open the ipynb.

* Local (TBU)

### Training
Run:
* `train_vit.ipynb` for Vision Transformer
* `train_effnet.ipynb` for EfficientNet

### Testing
Run `testing.ipynb`

### Troubleshooting
* For tensorboard errors: just remove all the tensorboard parts 
* For model retrieval errors (during testing): directly input the directory address of the model 
