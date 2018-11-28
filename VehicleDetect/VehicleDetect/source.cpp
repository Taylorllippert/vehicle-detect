#include <opencv2/core/core.hpp>
#include <opencv2/highhui/highgui.hpp>


/*
Reference for onjdetect.hpp

https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html
contains function to compute descriptor for image, 
contains some SVM functions,
contains detect functions


Constructor
HOGDescriptor(
	Size _winSize,
	Size _blockSize,
	Size _blockStride,
	Size _cellSize, 
	int _nbins, 
	int _derivAperature = 1, 
	double _winSigma = -1,
	int _histogramNormType = HOGDescriptor::L2Hys,
	double _L2HysThreshold = 0.2, 
	bool _gammaCorrection = false,
	int _nlevels = HOGDescriptor::DEFAULT _NLEVELS,
	bool _signedGradient = false)

Computes HOG descriptor of given image
virtual void HOGDescriptor:: compute(
	InputArry img,
	std::vector<float> &descriptors,
	Size winStride=Size(),
	Size padding = Size(),
	const std::vector <Point> &locations = std::vector<Point>()) const
*/
#include "objdetect.hpp"


/*double check configuration to compile for x64
 * add project property sheet from canvas */



using namespace cv;
