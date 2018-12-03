/* ---------------------------------------------------------------------------
 * Adam Ali, Taylor Brady
 * CSS 487 A
 * December 5, 2018
 * Program 4 Vehicle Detect
 * ---------------------------------------------------------------------------
 * Vehicle Detect is a project that explores object recognition techniques to 
 * locate vehicles in images, such as those from traffic and dash cams. Drawing 
 * a simple bounding box around each vehicle in an image is the objective.
 *
 * Vehicle Detect is written purely in C++ using OpenCV. In fact, both computer 
 * vision and machine learning concepts are applied here.
 *
 * We start with about 9K labeled images from Udacity. We want to build a simple 
 * model that has a basic understanding of what a “car” and “non-car” typically
 * look like. It turns out that the included labels are already “car” and 
 * “non-car”. Each training image is 64x64 pixels.
 *
 * In order for a computer to obtain this understanding, we need some 
 * quantitative representation of the images that can be more easily 
 * comparable. We achieve this using Histogram of Oriented Gradients. When we 
 * obtain this numerical profile of “cars” and “non-cars”, we need to find a 
 * way to plot these representations and use classification techniques to 
 * label new test images based on similarity to already labeled data. We 
 * achieve this using Support Vector Machine.
 * ---------------------------------------------------------------------------
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

const string CAR_PATH      = "dataset/car/*.png";
const string NONCAR_PATH   = "dataset/non-car/*.png";
const string SVM_PATH      = "dataset/svm.yml";
const string TEST_PATH     = "test.png";
const int    TRN_IMG_SIZE  = 64;
const int    PATCH_SIZE     = 8; 
const int    CAR           =  1;
const int    NONCAR        = -1;
const Scalar BOX_COLOR(255, 0, 0);
const int    BOX_THICKNESS = 4;

bool svm_exists(const string &path);
void read_dataset(vector<Mat> &dataset, const string &path);
void build_hogs(const vector<Mat> &dataset, vector<Mat> &histogram, const int &pixels);
void svm_train(const vector<Mat> &histogram, const vector<int> labels);
void detect_cars(const int &window);
void draw_box(Mat &scene, const vector<Rect> &positions);

/* svm_exists: checks if an SVM (yml) file already exists at path.
 * Preconditions: (path) is nonempty and is the assumed location the SVM data.
 * Postconditions: true if exists, otherwise false.
 */
bool svm_exists(const string &path) {
	ifstream f(path);
	return !f.good();
}

/* read_dataset: loads the training data.
 * Preconditions: (path) is nonempty.
 * Postconditions: (dataset) contains a series of matrices for all input.
 */
void read_dataset(vector<Mat> &dataset, const string &path) {
	vector<string> files;
	glob(path, files, false);

	Mat image;
	for (int f = 0; f < files.size(); f++) {
		image = imread(path.substr(0, path.length() - 6).append(files.at(f)));
		dataset.push_back(image.clone);
	}
}

/* build_hogs: builds a histogram of oriented gradients for every training image.
 * Preconditions: (dataset) is populated with training image matrices.
 *                (pixels) is the size of the training images, assumed square.
 * Postconditions: (histograms) contains a series of hogs for all input.
 */
void build_hogs(const vector<Mat> &dataset, vector<Mat> &histograms, const int &pixels) {
	HOGDescriptor hog;
	hog.winSize = Size(pixels, pixels);
	
	Mat grayscale;
	vector<float> magnitudes;
	vector<Point> orientations;
	for (int i = 0; i < dataset.size(); i++) {
		cvtColor(dataset.at(i), grayscale, COLOR_BGR2GRAY);
		hog.compute(grayscale, magnitudes, Size(PATCH_SIZE, PATCH_SIZE), Size(0, 0), orientations);
		histograms.push_back(Mat(magnitudes).clone());
	}
}

/* svm_train:
 * Preconditions:
 * Postconditions:
 */
void svm_train(const vector<Mat> &histograms, const vector<int> labels) {
	Ptr<SVM>  svm = SVM::create;
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1);
	svm->setC(0.01);
	svm->setType(SVM::EPS_SVR);
}

/* detect_cars:
 * Preconditions:
 * Postconditions:
 */
void detect_cars(const int &window) {

}

/* draw_box:
 * Preconditions:
 * Postconditions:
 */
void draw_box(Mat &scene, const vector<Rect> &positions) {
	for (int p = 0; p < positions.size(); p++) {
		rectangle(scene, positions.at(p), BOX_COLOR, BOX_THICKNESS);
	}
}

 /* main:
  * Preconditions: 
  * Postconditions: 
  */
int main(int argc, char* argv[]) {
	// do not train if already completed
	if (!svm_exists(SVM_PATH)) {
		// groups for each classification
		vector<Mat> cars;
		vector<Mat> noncars;

		// a histogram per image
		vector<Mat> histograms;

		// indices correspond to (histograms) vector, contains label
		vector<int> label_map;

		// build labeled dataset
		read_dataset(cars, CAR_PATH);
		read_dataset(noncars, NONCAR_PATH);
		label_map.assign(cars.size(), 1);
		label_map.insert(label_map.end(), noncars.size(), -1);

		build_hogs(cars, histograms, TRN_IMG_SIZE);
		build_hogs(noncars, histograms, TRN_IMG_SIZE);
		svm_train(histograms, label_map);
	}

	// display results of test scan
	detect_cars(TRN_IMG_SIZE);
}