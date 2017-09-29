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

[image1]: ./Distribution.jpg "Data Distribution"
[image2]: ./test_images/00018.jpg "Traffic Sign 1"
[image3]: ./test_images/00013.JPG "Traffic Sign 2"
[image4]: ./test_images/00014.jpeg "Traffic Sign 3"
[image5]: ./test_images/00035.jpg "Traffic Sign 4"


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
* Training set accuracy: 99.6724%
* Validation set accuracy: 97.8231%
* Testing set accuracy: 96.5162%

I used a simple architecture with 5 convolutional layers compared to the 2 convolutional layers in the LeNet architechture. As there were a large number of classes to classify (43) and many features in each classes to classify, a deep network is required and therefore I used 4 fully convoluted layers along with the convolutional layers. I avoided the overfitting by use of dropout in some layers after maxpooling and was able to obtain a validation accuracy of 97.8 %

5. Testing a Model on New Images

1. Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] 



2. Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      		| General Caution   									| 
| Yield     			| Yield 										|
| Stop 					| Stop											|
| Ahead Only	      		| Ahead Only					 				|
| Slippery Road			|      Slippery road 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

3. For the First and the Third images surprisingly the predictions came be a perfect 1 , that is mostly beacause of the rounding of to 3 decimal places. And also for the other images expect the 5th one it is obtained that the prediction probabilities are greater that 0.99. So it wouldn't be useful to comapre the softmax probabilities of the first four images and therefore I just included the 5th image softmax predictions  

For the Fifth image, the model is relatively sure that this is a Slippery road sign (probability of 0.986), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9864         			| Slippery road   									| 
| .0035     				|  				Wild animals crossing						|
| .0027					|   Bicycles crossing									|
| .0024	      			| 	 		Children crossing		|
| .0020				    |     Beware of ice/snow  							|

From the results above for the fifth image it shows that the network was able to classify the image as slippery road getting a probabilty of 98%.
From the results about and the rest depicted in the notebook itself it is quite evident that the model is performing well!
