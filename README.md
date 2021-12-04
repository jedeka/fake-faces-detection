# Fake Faces Detection -- 20hrs Challenge
---   
Two approaches for detecting fake faces (_e.g._ DeepFake) using EfficientNet and Vision Transformers. Due to local resource restrictions, Google Colab is utilized (with GPU support). 

Note that this is a timed challenge, thus the settings are not the most optimal (_e.g._ without hyperparameter tuning, data augmentation, etc.), and we ignore time features as the project is doing more image classification task rather than object detection and tracking.  

### Dataset
* [Click here for the split dataset](https://drive.google.com/drive/folders/1RrDFPuDWJtM-D8Tri_crTPpOmf_n0nT3?usp=sharing)  
(Split by sklearn's `train_test_split` with 0.8 : 0.2 train:test proportion. Using this data is recommended, to save time on data preprocessing)  
* [Click here for the full image dataset](https://drive.google.com/drive/folders/1TyjYmiyRoo7WQoqIX2X0P5zPfjTdUi5j?usp=sharing)  
(Use this data if you want to process the data more and tweak the splitting parameters. If you are using this dataset, you can use and modify `split_data.py` to split`and export the data)


### Setup
---   
* Google Colab  
To run this code inside Google Colab, clone this repo inside Colab by executing `!git clone https://github.com/sugaarrrr/fake-faces-detection.git` inside the notebook, or clone locally first and put the directory inside Google Drive afterwards.


* Local  
Using virtual environment (Python 3.7) is recommended. Once the virtual environment is created, run `!pip3 install -r requirements.txt`. 


### Requirements
---  
```
numpy
pandas
sklearn
ipython
tensorflow==2.2.0
torch
torchvision
keras
matplotlib
seaborn
jupyter
tensorflow-addons==0.10.0
efficientnet
einops
```

### Training
---   
Run:
* `train_vit.ipynb` for Vision Transformer
* `train_effnet.ipynb` for EfficientNet

### Testing
Run `testing.ipynb`

### Troubleshooting
* For model retrieval errors (during testing): directly input the directory address of the model 
