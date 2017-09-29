#**Traffic Sign Recognition** 
---

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Distribution.jpg "Data Distribution"
[image2]: ./test_images/00000.ppm "Traffic Sign 1"
[image3]: ./test_images/00016.ppm "Traffic Sign 2"
[image4]: ./test_images/00027.ppm "Traffic Sign 3"
[image5]: ./test_images/00033.ppm "Traffic Sign 4"
[image6]: ./test_images/00040.ppm "Traffic Sign 5"


1) Data Set Summary & Exploration:

I loaded the dataset from in the pickled form and segregated the images and the lables in different arrays for training data, validation  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

2) Exploratory visualization of the dataset:

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among all the classes for the three datasets.

![alt text][image1]

3) Design and Test a Model Architecture:
 
Preprocessing the dataset:
1) As a part of preprocessing the dataset I first converted the images into grayscale, the reason being the less number of parameters to handle in grayscale images compared to a rgb images where it will scale to 3x
2) Then I normalized the images to convert the pixel values to range between (-1 and 1) and thus making sure there is stability in the input to the network. 

Model Architecture: I used a custom convolutional layer architecture with 5 Convolutional layers and 4 Fully connected layers. 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 26x26x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 24x24x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x32 				|
| Dropout |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 				|
| Dropout |
| Flatten | output 1024 |
| Fully connected		|    outputs 512  					|
| Dropout |
| Fully connected		|    outputs 256  					|
| Dropout |
| Fully connected		|    outputs 128  					|
| Dropout |
| Fully connected		|   outputs n_classes   									|
|				Logits		|												
 
To train the model I used Adam optimizer and trained the model for 12 epochs with a batch size of 32 and dropout for the applicable layers as 0.6.

4. I started with the LeNet architecture but could not obtain an accuracy more than 90 % on average per epoch and then trianed the data with model of 9 layers - 5 Convolutional and 4 Fully connected and was able to obtain an accuracy of 98.2 % on validation data. I also used dropout in few layers to prevent the model from overfitting.

My final model results were:
 * Training set accuracy: 99.7787%
 * Validation set accuracy: 98.2540%
 * Testing set accuracy: 96.5717%

I used a simple architecture with 5 convolutional layers compared to the 2 convolutional layers in the LeNet architechture. As there were a large number of classes to classify (43) and many features in each classes to classify, a deep network is required and therefore I used 4 fully convoluted layers along with the convolutional layers. I avoided the overfitting by use of dropout in some layers after maxpooling and was able to obtain a validation accuracy of 98.2 %

5. Testing a Model on New Images

Here are five German traffic signs that I found on the web ( Actually taken from the online test data of the German competetion which is not a part of the downloaded):

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


