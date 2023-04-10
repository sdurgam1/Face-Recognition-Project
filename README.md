# Face-Recognition-Project
Face Recognition Model trained on celebrity images using Convolutional Neural Networks (CNN) </br></br>
<b>Introduction:</b></br>
<p align="justify">
Since the 1960s, face recognition has been a topic of interest. Over the years, technological advancements have enabled the enhancement and integration of this technology into everyday life. Face recognition is increasingly being used by law enforcement and security organizations in security surveillance and crime detection. This technology has been integrated into every aspect of our lives, from phone unlocking mechanisms to prevention or detection of identity theft.
</p>
<p align="justify">
Face recognition involves using an individual’s face in order to recognize and identify them. It has a wide variety of uses, ranging from employee identification in large corporations to student attendance in educational institutions. A face recognizer is made up of two parts. The first part involves a face detection method, which consists of finding faces in a photo or in a video stream, which often poses a challenge during the face detection. The second part involves verifying or identifying the person's identity. While there are numerous methods for achieving the same, this paper proposes a method to implement a Face Recognition system using machine learning which can be used in many day-to-day applications.
In this project, I have used the concept of HOG face detection to find the coordinates of the face in the image. This method of face detection is used to convert the image to grayscale and then draws the arrows that indicate the flow from lighter pixels to darker pixels, these arrows are called gradients. This process is useful to extract the actual features from the image and remove the distinction between dark and bright images of the same person. This helps in capturing the actual feature from the image. I have used the deep learning method called Convolutional Neural Network (CNN) to create face recognition model.
</p>
<b>Dataset</b></br>
<p align="justify">
For the dataset, I have used the face recognition dataset from Kaggle[1]. This dataset consists of 31 folders each of which represents a person who is a famous celebrity. Each of these folders contains 50 to 100 images of that celebrity. This dataset consists of a total amount of 2562 images and has a size of 726 MB. For analysis, I have excluded 2 folders from the data as both folders contain a single image of that celebrity.
</p>
<b>Concept of HOG:</b></br>
<p align="justify">
A feature extraction module named Histogram of Oriented Gradients (HOG) is frequently employed to extract facial features from the input passed to the algorithm. Other uses of this algorithm are in object detection in the field of computer vision. The HOG feature emphasizes an object's structure or shape and only evaluates if a pixel is an edge in the case of edge features. HOG also gives the edge direction. To achieve this, the gradient and orientation of the edges are extracted. Furthermore, "localized" components of these orientations are calculated. As a result, the entire image is divided into smaller patches, and the gradients and orientation are evaluated for each of those regions. In the end, the HOG would create a distinct Histogram for each of these zones. The term "Histogram of Oriented Gradients" refers to the histograms that are produced by the gradients and orientations of the pixel values
</p>

<b>WORKING OF HOG DETECTOR: </b></br>
<p align="justify">
HOG is a simple yet effective feature description. It is frequently used for item detection, including cars, animals, and fruits, in addition to face detection. Because HOG is a reliable objection detection technique that uses the local intensity gradient distribution and edge direction to characterize object shape.

<b>Step1:</b> In HOG, we divide the image into smaller cells.

<b>Step2:</b> Compute histogram for each cell using the gradients and orientations of the pixels.

<b>Step3:</b> Put these histograms together to create the feature vector, which creates a single, distinctive histogram for each face from all the little histograms.

I have tested the code on sample images.
</p>

<b><ins>Sample Inpt Image:</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Input%20sample%20image.jpg?raw=true" alt="sample input image missing" title="Sample Input Image">
</p>


<b><ins>HOG Ouput:</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Hog%20output%20image.jpg?raw=true" alt="sample input image missing" title="HOG output">
</p>

<b><ins>Ouput:</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/output%20image.jpg?raw=true" alt="sample input image missing" title="HOG output">
</p>

<b>Results</b></br>
<p align="justify">
I have tested the model on the test images, and the sample outputs observed are as follows.
</p>

<b><ins>Predictions of test images:</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Preictions%20of%20test%20images.jpg?raw=true" title="HOG output">
</p>


<b><ins>The confusion matrix for output data as follows:</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Confusion%20Matrix.jpg?raw=true" alt="sample input image missing" title="HOG output">
</p>

<p align="justify">
I have used the “accuracy” metric to evaluate the model. Accuracy gives the percentage of images that are correctly classified by the model. It is calculated by dividing the number of correctly classified predictions by the total number of predictions.
</p>
<p align="center">
		<b>Accuracy</b> =  (TP+TN)/(TP+TN+FP+FN) 
</p>
<p align="justify">
where True Positives (TP) is the number of images that are correctly classified. True Negatives (TN) is the number of images if correctly predicted it does not belong to a certain class. False Positives (FP) are the number of images that are incorrectly classified. False Negatives (FN) are incorrectly predicted that it does not belong to a certain class.
</p>

<b><ins>Acurracy Plot 1 (10 epochs) :</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Accuracy%20Plot%202.jpg?raw=true" alt="sample input image missing" title="HOG output">
</p>

<b><ins>Acurracy Plot 1 (25 epochs):</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Accuracy%20Plot%203.jpg?raw=true" alt="sample input image missing" title="HOG output">
</p>
<b><ins>Acurracy Plot 1 (50 epochs):</ins></b></br>

<p align="center">
<img src="https://github.com/sdurgam1/Face-Recognition-Project/blob/main/Results%20Images/Accuracy%20Plot%203.jpg?raw=true" alt="sample input image missing" title="HOG output">
</p>

<p align="justify">
The accuracy plot shows the accuracy and validation accuracy of the model with respect to epochs on the x-axis and accuracy percentage on the y-axis. We can see that the model was continuously increasing its accuracy for 8 epochs and then stayed the same with few fluctuations from the 8th epochs to the 50th epochs. From the plot, we can conclude that training the model beyond the 9th epoch was overfitting the model since the training accuracy was increasing but the validation accuracy was decreasing. The final model accuracy was 80.3%. I also performed an accuracy test on a completely new set of images and got a model accuracy of 80.4%.
</p>
