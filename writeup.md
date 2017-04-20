# Traffic Sign Recognition  
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

[image1]: ./report_images/dataset_capture.png "Dataset visualization"
[image2]: ./report_images/normalization.png "Normalization"
[image3]: ./report_images/values.png "Model during epochs"
[image4]: ./test_images/atention32.jpg "Traffic Sign 1"
[image5]: ./test_images/no_vehicles32.jpg "Traffic Sign 2"
[image6]: ./test_images/road_work32.jpg "Traffic Sign 3"
[image7]: ./test_images/stop32.jpg "Traffic Sign 4"
[image8]: ./test_images/trucks32.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/sorny92/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the shape of the arrays that contain the data to know the next information:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here we can see a photo of every label/class we can find in the data set and it's label on the top. 
Thanks to this captures we can see some of the features of the dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fith code cells of the IPython notebook.

I only used a simple normalization system to preprocess the images so it's easier for the NN to achieve a minima without overfitting.

Here is an example of two traffic signs before and after normalization.

![alt text][image2]

Then, in the sixth code cell I apply the preprocessing to the test dataset so it is in the same conditions as the dataset for training.  

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 9x9x65 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 9x9x65    				|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 9x9x130 	|
| RELU					|												|
| Fully connected		| output depth 160								|
| RELU					|												|
| Fully connected		| output depth 110								|
| RELU					|												|
| Fully connected		| output depth 55								|
| RELU					|												|
|						|												|
 
Which are basically adding a 1x1 convolution and another fully connected layer to a Lenet-5 architecture.

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an Adam optimizer with 30 epochs of training an a learning rate value of 1.5·E⁻3.

Batch size differed from the training environment. Firstly, I tested the architecture in an AWS machine so I was able to use high values of BATCH_SIZE as 2048, but when I trained in local I was only able to use a value of 64.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.968 (red line in image3)
* test set accuracy of 0.946

![alt text][image3]

I started working with a Lenet-5 architecture. I expected to achieve validation accuracies of 0.90 or higher, but I couldn't get higher than 0.4.
This was a problem due to a wrong configuration of variables in tensorflow. I forgot to set up a standard deviation of values that achieved low values. This created high parameters that avoided a minimum and underfitting the model.  

I added a fully connected layer and 1x1 convolution because underfitting meaned that not all the features were captures in the model. After this I achieved a validation accuracy around 0.84 (orange and blue lines in the above graph)  

Once I solved the problem with the weights initialization my validation accuracy rised to more than 0.92.  

I played with different learning rates till I achived an accuracy that satisfied me. (red, soft blue, green and dark red lines on the graph)

I tested to train the model for 200 epochs which gave me accuracy values of 0.975 but this was an extreme overfitting problem because in test situation I got 0.05 accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

First, second and fifth images seems easy to classify because light conditions are good and there's no oclusion or weird angles of the images.  
Second and third could be more tricky because the third is blury and the forth one has a unexpected dark ring around the sign that could trick the classifier.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twelfth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield           		| Yield     									| 
| No entry     			| No entry 										|
| Road works    		| General caution								|
| General caution  		| General caution				 				|
| 3.5 mt prohibited 	| 3.5 mt prohibited    							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 0.8. The number says my model would be overfitting because in test accuracy I get 0.96, but 5 images are not considered enough for statistical relevance. I would need more images to guess if it is working properly. That's why we have a test dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is quite sure about the classification because it gives a correct prediction with a value of 0.856.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.856 				| Yield											|
| 0.135 				| No vehicles									|
| 0.005 				| Roundabout mandatory							|
| 0.002 				| Keep right									|

For the second image, the model is totally sure about the classification because it gives a correct prediction with a value of 1.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| No entry										|
| 0.000 				| Speed limit (120km/h)							|
| 0.000 				| Traffic signals								|
| 0.000 				| Stop											|

For the third image, the model is quite sure about the classification because it gives a prediction with a value of 0.84 but this prediction is not correct so there is going to be need an evaluation of the model to see why the model is that certain about this sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.840 				| General caution								|
| 0.160 				| Go straight or left							|
| 0.001 				| Dangerous curve to the right					|
| 0.000 				| Road work										|

For the forth image, the model is sure about the classification because it gives a correct prediction with a value of 1.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| General caution								|
| 0.000 				| Speed limit (20km/h)							|
| 0.000 				| Speed limit (30km/h)							|
| 0.000 				| Speed limit (50km/h)							|

For the fifth image, the model is sure about the classification because it gives a correct prediction with a value of 1.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 				| Vehicles over 3.5 metric tons prohibited		|
| 0.000 				| Speed limit (80km/h)							|
| 0.000 				| Speed limit (100km/h)							|
| 0.000 				| Right-of-way at the next intersection			|
