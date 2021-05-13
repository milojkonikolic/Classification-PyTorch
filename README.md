# Classification-PyTorch

Implementation of the ResNet architectures for image classification.
ResNet architectures that are implemented are: ResNet18, ResNet34,
ResNet50, ResNet101, ResNet152.

## Data
Dataset consists of 102 classes of flowers and could be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
Number of images in the training part is about 7000 and number of images in the 
validation dataset is about 1000. </br>
For both training and validation datasets have to be generated json files - list 
of dictionaries where each element has keys "image_path" and "label". The example 
is shown below.
```
[
  {
    "img_path": "C:\\Machine Learning\\Datasets\\flowers_102\\data\\image_00001.jpg",
    "label": 76
  },
  {
    "img_path": "C:\\Machine Learning\\Datasets\\flowers_102\\data\\image_00008.jpg",
    "label": 76
  },
  ...
]
```
Also, class names have to be written in txt file where number of line
represents class id of that class name.
## Training
To start training run script train.py: ```python train.py --config config.yaml```. 
All parameters are written in the config.yaml file.
## Results

