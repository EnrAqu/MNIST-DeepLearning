# MNIST-DeepLearning

## Intro 

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning.Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset The original creators of the database keep a list of some of the methods tested on it.[6] In their original paper, they use a support-vector machine to get an error rate of 0.8%. An extended dataset similar to MNIST called EMNIST has been published in 2017, which contains 240,000 training images, and 40,000 testing images of handwritten digits and characters.

For simplification, images has been stored in csv file. The train.csv has 785 columns, the fist column is the label and the rest 784 contain the pixel value of the associated image pixel. For the python notebook, instead, we'll use the comand: `from tensorflow.keras.datasets import mnist`.

`train.csv` - (60000 samples) This csv file contains the pixel values as columns along with the digits it represent.
`test.csv` - (10000 samples) File that will be used for actual evaluation for the leaderboard score and it does not have the digit represented by the pixel values.

## File in repository 

The repository contains file in C, Python Notebook and Python, separated in folders. The `.csv` file are in a zip archive, and to run C neural net is necessary to create a binary file through `read_csv_to_binary.c`. The file without extensions are Unix Program, made by terminal with the command 

> gcc name_file.c -o name_program 

In this moment there is:

* CNN neural net in Jupyter Notebook;
* a binary file creator in `.c`;
* ANN neural net without stochastic gradient and one with it in C. 
