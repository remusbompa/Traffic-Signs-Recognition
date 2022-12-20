# Traffic-Signs-Recognition

## Overview

In this project, we created an application for the detection and classification of traffic signs in nature (in the wild), which uses the training results of 7 data sets, with an interactive graphic interface and some additional facilities, which helps to compare results and to identify possible performance problems of some datasets. For efficient image processing and extraction of detections with the highest possible accuracy, we used the method for which the best results were recorded in this field: deep neural networks, respectively convolutional networks. Among the existing convolutional architectures, YOLOv3 performed best on the COCO object recognition dataset, achieving accuracy performance similar to other models such as RetinaNet, but with 4x less inference time. We trained the network on the 7 most popular European datasets: GTSDB and GTSRB (from Germany), Swedish, Croatian, BTSD and BTSC (from Belgium) and a European dataset that gathers characters from all previous datasets.

After re-annotating the sets in YOLO format, we trained them for a certain number of iterations. For sufficiently trained datasets or with an average number of classes (below 70), we obtained similar (European, GTSRB and Swedish) or even better (BTSC) results compared to those reported in the reference articles. For detection within the application, we implemented the YOLO model in pytorch and applied its specific algorithms both at the input to the network: resizing images to fit the input size of the network, and at the output: object confidence thresholding and non- maximum-suppression. As the classification datasets (BTSC, GTSRB and European) focus more on traffic sign recognition in a region of interest, for them the image must be previously passed through a trained YOLO network for detection using one of the other datasets data. Following this optimization, we obtained better traffic sign classification results on the 3 classification sets compared to the other detection sets.

For tracking traffic signs in videos, we used 2 tracking algorithms: SORT and Deep SORT, which use Kalman filters for motion prediction and an algorithm for associating detections from the current step with previously detected objects, based on a distance. If in SORT, the metric used is IoU (intersection over union), in Deep SORT feature extraction with a Siamese network is used. I implemented both methods and trained this model for a number of epochs until I achieved acceptable accuracy so that I could use the resulting weights. We recorded good results, the signs being recognized from one frame to another (they are assigned the same track id), except for some cases where they are blocked for several seconds or where the sign has rotated a lot. Better performance was obtained for Deep SORT, which identified fewer unique objects while playing a video than the other simpler algorithm, SORT.

We have also improved the GUI so that it is accessible to the user and provides some additional functionality in addition to video and image detection. Thus, we implemented an interaction window with the server, through which the network can be trained on a data set chosen from the 7 available and the status of the two GPUs (busy or available) can be checked. We also built in an option to display comparative graphs between trained data sets or between classes of a given set, based on a chosen metric. In this way, the user is helped to analyze the differences between the data sets used and to notice possible errors that caused poorer performance for a certain data set.

## Conceptual schema

![Conceptual schema](/resource/schema_conceptuala.png)
*Conceptual schema*

In order to use the performance of the YOLO network in the case of traffic sign detection in nature (in the wild), we built a conceptual model of the application, which has this in the center, but also uses other tools to ensure the identification of signs between frames of the video stream and provides facilities for training and performance analysis of multiple data sets. Thus, the system has 4 functionalities: detecting signs in images, in videos, training datasets on the Server and displaying comparative graphs of the performance of several datasets.

The main objective of the project and the most difficult one is the classification of traffic signs from videos, this being made more difficult by the difficulty of following the signs in the conditions of a dynamic frame, the possible variations in brightness, the change of perspective, but also due to the need to provide the answer in a short time, so that the quality of the video stream is not affected. In this sense, for video processing, we considered a more complex chain and more interactions within the conceptual scheme: the frames of the video are passed one by one through the network for the detection of regions of interest and a superficial classification (in the case of detection sets), after which to achieve better classification performance, the obtained results are passed through YOLO once again (in the case of classification strings). If traffic sign tracking is not desired, the process can end at this point, displaying only the appropriate annotations. If the tracking option is selected, a specific algorithm will be called: SORT or Deep SORT, both based on Kalman filters for motion estimation. Deep SORT uses a neural network to extract features from detections that will be used to associate detections with already monitored objects (tracks), while SORT uses the IoU (intersection over union) metric. The result obtained consists in identifying the same objects by the same index as long as they remain in the frame. Thus, tracking worked better if fewer different objects were identified at the end of the clip.

Network training involves connecting to a server, which will save the results obtained in an xml file at the end of the execution. Using this file, which will be transferred to the user's local machine, he will be able to compare the results recorded on the available data sets and view various graphs and statistics, which will help him better understand the causes that determined an increase or decrease performance degradation when training the network.

## Architecture 

![Application's architecture](/resource/architecture.png)
*Application's architecture*

The application architecture is composed of 4 main packages: ServerPackage, TorchPackage, GUIPackage, StoragePackage, whose modules help to achieve 4 functionalities:
* detection on images
* detection in video
* model training
* display statistics about the trained data sets (precision, recall, f1_score, mean average precision)

The StoragePackage contains classes and methods for manipulating the local database:
* reading configuration files, names and weights
* reading additional information about traffic signs
* saving weights and statistics, taken from the server
For image detection and detection in video sequences, the detect_images and detect_video methods from the modules of the same name will be called. They will run on a separate thread from the GUI and will load and run the Darknet network defined in the model module, using methods defined in the utilities module. Loading the network calls methods in the StoragePackage to work with the network files in the local database for the chosen data set.
To train the network, the client will connect through a terminal to the server and train the chosen dataset remotely using one or both GPUs provided by the server. Information about training results on the server is managed by the ServerPackage: it manages the xmls resulting from training the model on the server, for each individual data set. I configured the application on the server so that at the end of the training, it stores in an xml file information about the performance of the set as a whole, such as: total number of detections, number of objects detected, precision, recall, f1 score, true positives , false positives, false negatives, average IoU, mean average precision or runtime, but also information about the performance of each class: average precision, precision, recall, average IoU, f1 score, true positives, true negatives, false positives.

The process of displaying statistics involves searching for the training results of selected data sets (out of the 7 provided) in the local database, or running the databases on the server if the results are not available locally (in case the database is not is found in the server, an error is returned)

TorchPackage handles the detection part of the application, which is all about working with files (in the model.py file): it builds the network (which it reads from a .cfg configuration file and creates as an object of the Darknet class, which extends nn.Module), load the weights (by calling a function load_weights to read the weights), read the name file (which contains the labels of the detected objects, their numbering must correspond to the annotations of the training images).

The GUIPackage handles the management of the user interface and the part after the detection of traffic signs: writing the obtained information into images and interacting with the user. It is divided into several windows (QMainWidget type), which can have a toolbar and each one has a central Qwidget attached, containing text, buttons, progress bar. The initial window is App, which has the ChooseActionWidget attached and by selecting an option you can move to the phase of detecting images, videos, training on a server or displaying statistics.

## Screenshots

![Main menu](/resource/main_menu.png)
*Main menu*

![Image selection menu](/resource/detect_images_menu.png)
*Image selection menu*

![Image detection menu](/resource/detect_images_2.png)
*Image selection menu*

![Video selection menu](/resource/detect_video.png)
*Video selection menu*

![Video detection menu sample](/resource/detect_video_2.png)
*Video detection menu sample*

![Video detection menu](/resource/detect_video_3.png)
*Video detection menu sample*

![Analytics menu](/resource/analytics_1.png)
*Analytics menu sample 1*

![Analytics menu](/resource/analytics_2.png)
*Analytics menu sample 2*

![Analytics menu](/resource/analytics_3.png)
*Analytics menu sample 3*

![Analytics menu](/resource/analytics_4.png)
*Analytics menu sample 4*

![Initial train menu](/resource/train_menu.png)
*Initial train menu*

![Train menu while training](/resource/train_menu_2.png)
*Train menu while training*
