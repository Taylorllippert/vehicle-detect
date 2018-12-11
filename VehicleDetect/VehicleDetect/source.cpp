/* ---------------------------------------------------------------------------
 * Adam Ali, Taylor Brady
 * CSS 487 A
 * December 5, 2018
 * Program 4 Vehicle Detect
 * ---------------------------------------------------------------------------
 * PLACE TEST IMAGES IN WORKING DIRECTORY. EDIT (TEST_PATH).
 * SVM ALREADY PROVIDED. DELETE dataset\svm.yaml TO RETRAIN.
 *
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
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

const string WINDOW_LABEL    = "VEHICLE DETECT";
const string CAR_PATH        = "dataset\\car\\*.png";
const string NONCAR_PATH     = "dataset\\non-car\\*.png";
const string SVM_PATH        = "dataset\\svm.yml";
      string TEST_PATH       = "test.png";
const int    TRN_IMG_SIZE    = 64;
const int    PATCH_SIZE      = 8; 
const int    CAR             =  1;
const int    NONCAR          = -1;
const float  HIT_THRESHOLD   = 0.0302;
const float  WINDOW_SCALING  = 0.6;
const float  FINAL_THRESHOLD = 2.0;
const int    BOX_THICKNESS   = 2;
const Scalar BOX_COLOR(0, 0, 255);
const Size   WINDOW_STRIDE(9, 9);
const Size   WINDOW_PADDING(5, 5);


bool on_disk(const string &path);
void read_dataset(vector<Mat> &dataset, const string &path);
void build_hogs(const vector<Mat> &dataset, vector<Mat> &histogram, const int &pixels);
void svm_train(const vector<Mat> &histograms, const vector<int> labels);
void svm_transpose(Mat &statistics, const vector<Mat> &histograms);
void svm_hyperplane(const Ptr<SVM> &svm, vector<float> &hyperplane);
void detect_cars(const int &window);
void draw_boxes(const vector<Rect> &positions, Mat &scene);

/*
 * on_disk: checks if file is present at (path).
 * Preconditions: (path) is valid.
 * Postconditions: true if file exists, otherwise false.
 */
bool on_disk(const string &path) {
	ifstream file(path);
	return file.good();
}

/*
 * read_dataset: loads data into a (Mat) data structure from (path).
 * Preconditions: (path) is valid.
 * Postconditions: (dataset) is loaded with all images from (path).
 */
void read_dataset(vector<Mat> &dataset, const string &path) {
	// use regex in global variable to generate all subfiles
	vector<string> files;
	glob(path, files, false);

	Mat image;
	for (int f = 0; f < files.size(); f++) {
		printf("\rREADING IMAGES %d/%d", f, files.size());
		image = imread(files.at(f));
		dataset.push_back(image.clone());
	}
	std::cout << " DATASET LOADED." << endl;
}

/*
 * build_hogs: computes histogram of oriented gradients for every image in (dataset).
 * Preconditions: (pixels) is the desired patch size.
 * Postconditions: (histogram) contains hogs for all images in (dataset).
 */
void build_hogs(const vector<Mat> &dataset, vector<Mat> &histogram, const int &pixels) {
	// specify patch size and bins for orient/magnitude
	HOGDescriptor hog;
	hog.winSize = Size(pixels, pixels);
	Mat grayscale;
	vector<Point> orientation;
	vector<float> magnitude;

	// build hogs based off gray-scale images
	for (int d = 0; d < dataset.size(); d++) {
		printf("\rBUILDING HOG %d/%d", d, dataset.size());
		cvtColor(dataset.at(d), grayscale, COLOR_BGR2GRAY);
		hog.compute(grayscale, magnitude, Size(PATCH_SIZE, PATCH_SIZE), Size(0, 0), orientation);
		histogram.push_back(Mat(magnitude).clone());
	}
	std::cout << " HOGS COMPLETE." << endl;
}

/*
 * svm_train: plots all hogs in (histograms) in the svm to facilitate
 *            calculating a decision function.
 * Preconditions: (histograms) contains the hogs of the entire dataset,
 *                positive and negative.
 *                (labels) has a 1:1 mapping with (histograms) such that
 *                each set of corresponding elements is a hog-label pair.
 *                that is, each hog has a positive/negative classification.
 *                this is supervised learning.
 * Postconditions: svm plot output to disk in .yml format. contains the
 *                 support vectors necessary to calculate the hyperplane.
 *                 written to (SVM_PATH).
 */
void svm_train(const vector<Mat> &histograms, const vector<int> labels) {
	// set SVM defaults
	std::cout << "Configuring SVM defaults... ";
	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1);
	svm->setC(0.01);
	svm->setType(SVM::EPS_SVR);
	std::cout << "Done." << endl;

	// transpose as needed to determine decision function
	std::cout << "Adjusting data for hyperplane calculation (transpose)... ";
	Mat statistics;
	svm_transpose(statistics, histograms);
	std::cout << "Done." << endl;

	// compute and output support vectors to disk
	std::cout << "Calculating support vectors... ";
	svm->train(statistics, ROW_SAMPLE, Mat(labels));
	std::cout << "Done." << endl;
	svm->save(SVM_PATH);
	std::cout << "SVM saved to " << SVM_PATH << endl;
}

/*
 * svm_transpose: transposes hogs if required in order to process
 *                hyperplane calculation.
 * Preconditions: (statistics) is empty, (histograms) has all hogs.
 * Postconditions: (statistics) is identical to (histograms), apart
 *                 from the hogs that needed transposing.
 */
void svm_transpose(Mat &statistics, const vector<Mat> &histograms) {
	int rows = histograms.size();
	int cols = max(histograms[0].cols, histograms[0].rows);

	Mat transposed(1, cols, CV_32FC1);
	statistics = Mat(rows, cols, CV_32FC1);
	
	for (int sample = 0; sample < histograms.size(); sample++) {
		if (histograms.at(sample).cols == 1) {
			transpose(histograms.at(sample), transposed);
			transposed.copyTo(statistics.row(sample));
		}
		else if (histograms.at(sample).rows == 1) {
			histograms.at(sample).copyTo(statistics.row(sample));
		}
	}

}

/*
 * svm_hyperplane: uses the support vectors in (svm.yml) to calculate
 *                 a hyperplane or decision function.
 * Preconditions: (svm_train) has been run such that (svm.yml) exists
 *                at (SVM_PATH).
 *                (svm) is the svm loaded from (svm.yml).
 * Postconditions: (hyperplane) contains the decision boundary.
 */
void svm_hyperplane(const Ptr<SVM> &svm, vector<float> &hyperplane) {
	Mat support_vectors = svm->getSupportVectors();
	int num_vectors = support_vectors.rows;
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	hyperplane.resize(support_vectors.cols + 1);
	memcpy(&hyperplane[0], support_vectors.ptr(), support_vectors.cols * sizeof(hyperplane[0]));
	hyperplane[support_vectors.cols] = (float)-rho;
}

/*
 * detect_cars: uses a sliding window to classify windows within the 
 *              calculated svm/hyperplane.
 * Preconditions: (window) is equal to (TRN_IMG_SIZE).
 * Postconditions: bounding boxes drawn around matches.
 */
void detect_cars(const int &window) {
	// define sliding window search
	Mat scene = imread(TEST_PATH);
	HOGDescriptor patch;
	patch.winSize = Size(window, window);
	vector<Rect> boxes;

	// calc hyperplane/decision f unction
	Ptr<SVM> svm;
	svm = StatModel::load<SVM>(SVM_PATH);
	vector<float> hyperplane;
	svm_hyperplane(svm, hyperplane);
	patch.setSVMDetector(hyperplane);

	// define thresholds and movement of sliding window
	// fine tuning by trial-and-error
	patch.detectMultiScale(scene, boxes, HIT_THRESHOLD, WINDOW_STRIDE, WINDOW_PADDING, WINDOW_SCALING, FINAL_THRESHOLD, false);

	// render results
	draw_boxes(boxes, scene);
	imshow(WINDOW_LABEL, scene);
	waitKey(0);
	
}

/*
 * draw_boxes: draws boxes where specified.
 * Preconditions: (positions) contains upper-left and lower-right
 *                xy pairs.
 *                (scene) is the image to draw the boxes on.
 * Postconditions: (scene) has boxes drawn for all locations in
 *                 (positions).
 */
void draw_boxes(const vector<Rect> &positions, Mat &scene) {
	for (int p = 0; p < positions.size(); p++) {
		rectangle(scene, positions.at(p), BOX_COLOR, BOX_THICKNESS);
	}
}

 /* main: trains svm if no (svm.yml) exists. displays results regardless.
  * Preconditions: none.
  * Postconditions: detected vehicles according the svm understanding
  *                 are rendered with bounding boxes.
  */
int main(int argc, char* argv[]) {
	// grab test
	if (argc > 1) {
		TEST_PATH = argv[1];
	}

	// welcome msg
	std::cout << "+---------------------------------+" << endl;
	std::cout << "|     Adam Ali, Taylor Lippert    |" << endl;
	std::cout << "|         CSS 487 A Olson         |" << endl;
	std::cout << "|        December 5, 2018         |" << endl;
	std::cout << "|          VEHICLE DETECT         |" << endl;
	std::cout << "+---------------------------------+" << endl;

	// verify dataset is present
	bool cars_on_disk = on_disk(CAR_PATH.substr(0, 12) + "1.png");
	if (cars_on_disk) std::cout << "| Cars     at dataset\\car.......Y |" << endl;
	else              std::cout << "| Cars    at dataset\\car.......N |" << endl;
	bool noncars_on_disk = on_disk(NONCAR_PATH.substr(0, 16) + "extra1.png");
	if (noncars_on_disk) std::cout << "| Non-cars at dataset\\non-car...Y |" << endl;
	else                 std::cout << "| Non-cars at dataset\\non-car...N |" << endl;

	// check if svm was already trained
	bool svm_on_disk = on_disk(SVM_PATH);
	if(svm_on_disk) std::cout << "| SVM      at dataset\\svm.yml...Y |" << endl;
	else            std::cout << "| SVM      at dataset\\svm.yml...N |" << endl;

	// check if test image is present
	bool test_on_disk = on_disk(TEST_PATH);
	if (test_on_disk) std::cout << "| Test img at TEST_PATH.........Y |" << endl;
	else            std::cout << "| Test img at TEST_PATH.........Y |" << endl;
	
	std::cout << "+---------------------------------+" << endl << endl;

	// cannot proceed without a test image
	if (!test_on_disk) {
		std::cout << "TEST_PATH invalid. Nothing to test. Abort." << endl;
		return -1;
	}

	// cannot proceed without data if training is needed
	if (!svm_on_disk && (!cars_on_disk || !noncars_on_disk)) {
		std::cout << "SVM has not been trained. Dataset is incomplete. Abort." << endl;
		return -1;
	}

	if (svm_on_disk) {
		std::cout << "SVM has already been trained!" << endl;
	}

	// do not train if already completed
	if (!svm_on_disk) {
		std::cout << "SVM has not been trained. Training (this might take a while)..." << endl << endl;

		// groups for each classification
		vector<Mat> cars;
		vector<Mat> noncars;

		// a histogram per image
		vector<Mat> histograms;

		// indices correspond to (histograms) vector, contains label
		vector<int> label_map;

		// build labeled dataset
		std::cout << "LOAD CARS DATASET" << endl;
		read_dataset(cars, CAR_PATH);
		std::cout << endl << "LOAD NONCARS DATASET" << endl;
		read_dataset(noncars, NONCAR_PATH);
		label_map.assign(cars.size(), CAR);
		label_map.insert(label_map.end(), noncars.size(), NONCAR);

		std::cout << endl << "PROCESS CARS DATASET" << endl;
		build_hogs(cars, histograms, TRN_IMG_SIZE);
		std::cout << endl << "PROCESS NONCARS DATASET" << endl;
		build_hogs(noncars, histograms, TRN_IMG_SIZE);

		std::cout << endl << "BUILD SVM" << endl;
		svm_train(histograms, label_map);
	}

	// display results of test scan
	std::cout << endl << "Displaying results..." << endl;
	detect_cars(TRN_IMG_SIZE);
	std::cout << "Done." << endl;
	return 0;
}