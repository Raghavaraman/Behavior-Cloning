# Behavioral Cloning
##### Udacity Project
---
### Files:

The following are added to this git repo:
    1) drive.py
            Given by udacity to drive the car. Adjusted the speed to 10 for a proper smooth ride
    2) model.py
            The model I have developed to solve this particular problem
    3) model.h5
            The H5 for the successful neural network
    4) Preprocess.py
            Since I had few preprocesses on both drive and model python files I have it alone so that I can import it at both location
    5) README.Md
            Contains the markdown for the project.
            
### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

* python drive.py model.h5

### Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file is modularized and is easy to understand.

## Model Architecture and Training Strategy

### An appropriate model architecture has been employed

I have a used a simple model with 2 convolutional layer and 1 fully connected layer. This simple CNN is enough to run the car on the track.

### Attempts to reduce overfitting in the model

Since the CNN was a simple one with 3 hidden layers, I added only one dropout layer to redyce overfitting of the model. I have also used different data for training and validation.

### Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually.

### Appropriate training data

Training data was a combination of the Udacity data and my driving data. Training data was chosen to keep the vehicle driving on the road. I used a combination of images from all the 3 camera in the car.

### Solution Design Approach

Since I had a low power system, I decided to make the most out of the Udacity data and add just a few images from my own driving data set.

I created a simple CNN with 2 convolutions and 1 fully connected as I thought that would be more than sufficient to keep the car on the track. The kernel sizes were choose appropiately so that the CNN can easily detect the curves of the road in the first or the second convolution.

I selected images randomly from both the udacity data and my own data and split them into training and validation data. I got a validation loss of around 0.02 which i believe is sufficient to run the car on the track.

 By combining all these I was able to run the car on the track.
 
 ### Final Model Architecture
 
 My CNN consists of 2 Convolutional layers, the first of them has a kernel size of 5*5 so the its easy for it to find the lines/ends of the roads and road curves. I have also used filters of size 32,16 and a max pooling of (2,2).
 
 A dense layer of 64 and dropout is added to avoid overfiting before the output layer of 1.
 
 ### Creation of the Training Set & Training Process
 
 Since my PC wasn't capable of recording high quality data. I depended mostly on the Udacity data. I did add limited amount of my data too. The training set and validation set was choosen at random. I did have to record the last couple of curves twice just to be carful around them.
 
 ### Final simulation
 
 The car was able to run around the track with all the wheels staying on the road and without much of a deviation.




