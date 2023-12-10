# Image Classification - Rock Paper Scissors

**🎲General information**

[ Game 'rock paper scissors' described on Wikipedia](https://en.wikipedia.org/wiki/Rock_paper_scissors)

**Goal of this project** 

Classify images of hands showing gestures of the game 'rock', 'paper', or 'scissors' to classify the gesture as rock, paper, or scissors gesture.


**Dataset**

The dataset used in this project is available in the **TensorFlow datasets catalog**.

- [https://www.tensorflow.org/datasets/catalog/rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)

The following example images were taken from this dataset. It can be seen that the dataset contains hands in various skin tones capturing wide variety of hand shapes and orientations. The images are `300x300` pixels in size and are in color images (RGB). 
The images seem all to have a bright background and the hands are in the center of the images. 
Furthermore the hands seem always to be fully visible in the images, but seem to have different scales. 

![rock_paper_scissors_examples.png](rock_paper_scissors_examples.png)

The dataset itself only contains `test` and `train` splits. 

```	
Number of training (full train) examples: 2520
Number of test examples                 : 372
Ratio test data to (full) train data    : 14.76%
```

When comparing the amount of test data to the amount of training (here referred to as 'full train') data we can see that it is approximately 15% of the training data. Therefore the `train` split is further split into `train` and `validation` splits using 15% of the `train` split for creating the `validation` split.

## EDA and Model Training

Please refer to the notebook [./eda-training/README.md](./eda-training/README.md) for how to set up the development environment for running the EDA and Model Training.

Content covered in [./eda-training/README.md](./eda-training/README.md) 

1. Environment setup using Docker container
1. Exploratory Data Analysis (EDA)
1. Model Training



