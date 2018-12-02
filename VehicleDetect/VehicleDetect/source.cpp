#include <iostream>
#include <vector>
#include <fstream>
#include <string>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/detection_based_tracker.hpp>
#include <opencv2/objdetect/objdetect_c.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace cv::ml;

string svmFilename = "detector.yml";
#define TRUE 1
#define FALSE 0
#define DATA_SIZE 5

/*
Reference for onjdetect.hpp

https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html
contains function to compute descriptor for image, 
contains some SVM functions,
contains detect functions
*/


/* checkIfNeeded()
Purpose:
if this function has already been run,
	we don't need to do it again
so, this function checks if the file
	that would be the result has already been created
Precondition:
The string filename has been targeted for the
	file we want to check
Postcondition:
	True if the file doesn't exist
	False if the file does
*/
bool checkIfNeeded(string filename) {
	bool fileExists = false;
	std::ifstream f(filename);
	return !f.good();
}


/* Preform_Data_Analysis()
Purpose:
	Loads images and labels them accordingly
Precondition:
	Images of cars are located in the file
		../images/vehicles
	Images not of cars are located in the file
		../images/non-vehicles
PostCondition:
	A vector of images, and a vector of labels
		a car is labeled TRUE
*/
void Preform_Data_Analysis(vector<Mat> &img_lst, vector<int> &labels) {
	vector<String> car_files;
	vector<Mat> img_1st;
	glob("../images/vehicles/", car_files);
	vector<String> not_files;
	glob("../images/non-vehicles/", not_files);
	int carSize = (int)car_files.size();
	int notSize = (int)not_files.size();
	if (DATA_SIZE != 0) {
		carSize = DATA_SIZE;
		notSize = DATA_SIZE;
	}
	for (int i = 0; i < carSize; i++) {
		Mat input = imread(car_files[i]);
		if (input.empty())
			continue;
		img_lst.push_back(input);
		labels.assign(i, TRUE);
	}
	for (int i = 0; i < notSize; i++) {
		Mat input = imread(not_files[i]);
		if (input.empty())
			continue;
		img_lst.push_back(input);
		labels.assign(i + car_files.size(), FALSE);
	}


}


/* setUpSVM()
Purpose:
	Sets up an SVM with the desired parameters
Precondition:
	None
PostCondition:
	A pointer to the svm is returned
Why is this a function:
	It makes these parameters easy to change
	without setting them all as global variables
*/
Ptr<SVM> setUpSVM() {
	Ptr<SVM> svm = SVM::create();
	int kType = SVM::LINEAR;
	int svmType = SVM::EPS_SVR;

	/*
	LINEAR:		no mapping is done, linear discrimination is done in the original feature space. fastest
		K(x,y)=x'*y
	POLY
		K(x,y) = (gamma*x'*y + coef0)^degree, gamma > 0
	RBF:		radial basis function
		K(x,y) = e^(-gamma ||x-y||^2), gamma > 0
	SIGMOID
		K(x,y)=tanh(gamma*x'*y+coef0)
	CHI2:
		K(x,y)=e^(-gamma*z(x,y))
		z(x,y)=(x-y)^2/(x+y), gamma > 0
	INTER:		histogram intersection - fast
		K(x,y) = min(x,y)
	*/
	svm->setKernel(kType);
	//Kernel Function Params
	svm->setGamma(0);	//coeficient
	svm->setCoef0(0.0);	//constant 
	svm->setDegree(3);	//degree of polynomial

	/*
	C_SVC:		n-class classification, allows (imperfect) separation of classes with penalty multiplier C for outliers
	NU_SVC:		n-class classification, with possible imperfect separation. v is used instead of C. (the larger the value the smoother the decision boundary
	ONE_CLASS	1 class, separates the class from the rest of the feature space
	EPS_SVR:	epsilon-support vector regression. The distance between feature vectors from the training set and the fitting hyper-plane must be less than p. For outliers the penalty multiplier C is used
	NU_SVR:		v is used instead of p
	*/
	svm->setType(svmType);
	//SVM params
	svm->setC(0.01);	//penalty for multiplier
	svm->setNu(0.5);	//smoothness of boundary
	svm->setP(0.1);		//max distance between feature vectors and fitting hyper-plane

	/*
	sets tolerance and/or max iterations
	*/
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3));
	return svm;
}


/* Preform_SVM_Training()
Purpose:
	Trains the SVM network to recognize cars
Precondition:
	The input vector contains hog descriptors and the corresponding labels are in the label vector
PostCondition:
	SVM coefficients are returned as an input array
*/
vector<float> Preform_SVM_Training(vector<Mat> &input, vector<int>labels) {
	int R = (int) input.size();
	int C;
	if (input[0].rows > input[0].cols)
		C = input[0].rows;
	else
		C = input[0].cols;

	Mat trainData(R, C, CV_32FC1);
	for (int i = 0; i < R; i++) {
		input[i].copyTo(trainData.row(i));
	}

	vector<float>svmVec;
	Ptr<SVM> svm= setUpSVM();
	svm->train(trainData, ROW_SAMPLE, labels);

	Mat supportVectors = svm->getSupportVectors();
	Mat alpha, svidx;
	double r = svm->getDecisionFunction(0, alpha, svidx);
	svmVec=vector<float>(supportVectors.rows + 1);
	memcpy(&svmVec[0], supportVectors.ptr(), supportVectors.cols * sizeof(svmVec[0]));
	svmVec[supportVectors.cols] = (float)-r;
	return svmVec;
}


int main(int argc, char**argv) {
	HOGDescriptor h;
	//If the SVM has been trained we can skip this section
	if (checkIfNeeded(svmFilename)) {
		vector<Mat> trainingSet;
		vector<int> labels;
		Preform_Data_Analysis(trainingSet, labels);

		//Hog here
		

		h.setSVMDetector(Preform_SVM_Training(trainingSet, labels));
		h.save(svmFilename);
	}
	else {
		HOGDescriptor h;
		h.load(svmFilename);
	}


	Mat searchImage;
	if (argc > 1) {
		cout << argv[1] << endl;
	}
	else {
		searchImage = imread("search.jpg");
	}
	vector<Rect> locations;
	h.detectMultiScale(searchImage, locations);


	return 0;
}


