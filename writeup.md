# **Traffic Sign Recognition**

## Writeup



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./TrafficSignExamples.JPG "Traffic Sign Examples"
[image10]: ./TrafficSignDistribution.JPG "Traffic Sign Distribution"
[image11]: ./myGrayScale.JPG "Gray scale image"
[image12]: ./ImageRotate.JPG "Rotated Image"
[testimage1]: ./test_pics/EndLimits.JPG "End of All Limits"
[testimage2]: ./test_pics/Limit30.JPG "Limit 30km/h"
[testimage3]: ./test_pics/Limit120.JPG "Limit 120km/h"
[testimage4]: ./test_pics/NoPassing.JPG "No Passing"
[testimage5]: ./test_pics/Priority.JPG "Priority Road"
[testimage6]: ./test_pics/STOP.JPG "Stop"
[testimage7]: ./test_pics/Yield.JPG "Yield"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are examples of the 43 types of traffic signs:

![alt text][image9]


Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the 43 types of traffic signs.

![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the signs are not distinguishable by their color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image11]

As a last step, I normalized the image data to the range from 0.1 to 0.9 because this is numerically more stable.

I decided to generate additional data because otherwise the neural network would concentrate on learning the traffic signs that are more common in the training set. This can lead to overfitting.
Adding additional data acts as a kind of regularization.

To add more data to the the data set, I rotated signs from the same traffic sign class by an random angle in the range from -20 to 20 degree.

Here is an example of an original image and an additional rotated image:

![alt text][image12]

The difference between the original data set and the augmented data set is the following distribution of the traffic signs.
Each type has the same number of training data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride,  valid padding, outputs 10x10x16    									|
|RELU   |   |
|Max pooling   | 2x2 strides, outputs 5x5x16   |
| Flatten		| outputs 400 fully connected      									|
| Fully connected				| input 400, ouput 120        									|
|			RELU			|												|
|	Dropout					|												|
|Fully connected   | input 120, ouput 84   |
|RELU   |   |
|Fully connected   | input 84, output 43   |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. The training accuracy was much better than the validation accuracy which leads to the assumption that the solution suffers from overfitting. So I used the L2-regularization with weight beta to overcome this problem.

For the hyperparameters I choose:
- EPOCHS = 30
- BATCH_SIZE = 150
- beta = 0.005
- rate = 0.0005

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.943
* test set accuracy of 0.925

If a well known architecture was chosen:
* What architecture was chosen?

I did choose the LeNet architecture.

* Why did you believe it would be relevant to the traffic sign application?

The LeNet architecture was designed to learn grayscale images.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Without any improvements the training accuracy was quit high but the validation accuracy was very low (about 0.88). This lead me to the assumption that the model is overfitting the data. I tried turning the images to grayscale and normalizing it which improved the performance a bit.
Furthermore, the model has difficulty to learn the images which are scarce in the training set.
So I added additional data to the training set, by rotating some images from the existing training set.
This improved the performance a lot.
As there was still a gap between the accuracy of the training set and the validation set, I used an L2-regularization.

There is still a difference in the accuracy between the training set and the validation set.
I could not improve the accuracy of the validation set by increasing the regularization parameter.
The accuracy of the test set is almost the same as the accuracy of the validation set.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][testimage1]

This image (End of all limits) should be easy to  classify due to good lighting, no dirt, no refelctions, ... .

![alt text][testimage3]

This image (Speed limit 120km/h) should also be easy to classify.

![alt text][testimage2]

This image (Speed limit 30 km/h) should be easy to classify.

![alt text][testimage4]

This image (No passing) should be easy to classify although there are some reflections.

![alt text][testimage5]

This image (Priority road) should be easy to classify.

![alt text][testimage6]

This image (Stop) could be a little harder to classify as it is tilted and there are some refelctions on the sign.

![alt text][testimage7]

This image (Yield) could be a little harder to classify as it is a little bit tilted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| End of All Limits      		| End of All Limits   									|
| Speed limit 120 km/h     			| Speed limit (80km/h) 										|
| Speed limit 30 km/h					| Speed limit 30 km/h											|
| No Passing	      		| No Passing					 				|
| Priority road			| Priority road      							|
|Stop   | Stop   |
| Yield   | Yield   |


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 92.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

 The top five soft max probabilities were

| Image | 1. Probability/Sign | 2. Probability/Sign | 3. Probability Sign | 4. Probability/Sign | 5. Probability/Sign |
|:---------------------:|:---------------------------------------------:|
| End of all Limits | 99.9999% / End of all limits  | 0.0001% / End of no passing | 0.0% / End of speed limit (80km/h) | 0.0 % / Go straight or right | 0.0% / No entry
| Speed limit 120 km/h | 99.2639% / Speed limit (80km/h) | 0.6618% / Speed limit (30km/h) | 0.0306% / Speed limit (50km/h) | 0.016% / Speed limit (70km/h) | 0.0082% / Stop |
| Speed limit 30 km/h	| 99.9997% / Speed limit (30km/h)  | 0.0003% / Speed limit (50km/h)  | 0.0% / Speed limit (70km/h) | 0.0% / Speed limit (20km/h) | 0.0% / Speed limit (80km/h) |
| No Passing | 98.6313% / No passing | 1.3611% / No passing for vehicles over 3.5 metric tons | 0.0041% / Keep left | 0.0008% / Priority road | 0.0007% / Roundabout mandatory |
| Priority road	| 100.0% / Priority road | 0.0% / Keep right | 0.0% / Roundabout mandatory  | 0.0% / Yield | 0.0% / End of no passing |
|  Stop |  98.8084% / Stop | 0.6024% / Keep right | 0.2119% / Go straight or right | 0.199% / Speed limit (70km/h) | 0.0836% / Traffic signals |
| Yield  | 100% / Yield | 0.0% / Ahead only | 0.0% / Keep right | 0.0% / Turn right ahead | 0.0% / Go straight or right |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I did not complete the optional task.
