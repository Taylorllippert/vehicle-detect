**Should only run once unless parameters change**
Analysis: Divide data into training, validation, and then larger images
--> Input:   Images of cars, and not cars
--> Output:  2 sets of data in the same format, and trimmed for only necessary info
**Should only run once/image unless parameters change**
feature extraction: On training and validation
--> Input:   set of images
--> Output:  data set with HOG related to label

**Should only run once unless parameters change**
We need to train the SVM classifier
--> Input:    Training data set of HOG and labels, Validation data set of HOG and labels
--> Output:   Trained Network

**Performed for each large image**
Sliding Window Search : Preform HOG on each window
--> Input:    Image with multiple cars and other objects, SVM network
    feature extraction: On data within the window
    --> Input:    Partial image
    --> Output:   Hog descriptor
    Decision: Use trained network to determine if the image contains a car
    --> Input:    HOG descriptor, trained Network
    --> Output:   Boolean (yes it is a car or no it is not)
--> Output:   Location of 'cars'

HeatMap & Smooth : 
--> Input: List of locations of cars   
--> Output: Locations of cars based on which locations pass a threshold for car locations

place boxes


Functions
HOG extractor -- Adam
SVM           -- Taylor

Sliding Window -- Taylor
Heatmap        -- Adam

Analysis      -- Taylor
Desicion      -- Adam
