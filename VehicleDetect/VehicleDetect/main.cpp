// main.cpp
// This code demonstrates accessing pixels in OpenCV using different techniques
// Author: Clark Olson

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


// sharpen1 - sharpens an image by taking a weighted average of each pixels with its neighbors
// preconditions: image is an greyscale or color byte image
// postconditions: a new image with the same size is returned after sharpening
Mat sharpen1(const Mat &image) {
	Mat result = image.clone();

	if (image.channels() == 1) {
		for (int r = 1; r < image.rows - 1; r++) {
			for (int c = 1 ; c < image.cols - 1; c++) {
				result.at<uchar>(r, c) = saturate_cast<uchar>( 5 * image.at<uchar>(r, c)
										- image.at<uchar>(r+1, c) - image.at<uchar>(r, c+1)
										- image.at<uchar>(r-1, c) - image.at<uchar>(r, c-1) );
			}
		}
	}

	if (image.channels() == 3) {
		for (int r = 1; r < image.rows - 1; r++) {
			for (int c = 1 ; c < image.cols - 1; c++) {
				for (int b = 0; b < 3 ; b++) {
					result.at<Vec3b>(r, c)[b] = saturate_cast<uchar>( 5 * image.at<Vec3b>(r, c)[b]
										- image.at<Vec3b>(r+1, c)[b] - image.at<Vec3b>(r, c+1)[b]
										- image.at<Vec3b>(r-1, c)[b] - image.at<Vec3b>(r, c-1)[b] );
				}
			}
		}
	}

	return result;
}


// sharpen2 - sharpens an image by taking a weighted average of each pixels with its neighbors
// preconditions: image is an greyscale or color byte image
// postconditions: a new image with the same size is returned after sharpening
Mat sharpen2(const Mat &image) {
	Mat result = image.clone();

	if (image.channels() == 1) {
		for (int r = 1; r < image.rows - 1; r++) {
			const uchar * previous = image.ptr<const uchar>(r - 1);
			const uchar * current = image.ptr<const uchar>(r);
			const uchar * next = image.ptr<const uchar>(r + 1);

			uchar *output = result.ptr<uchar>(r);

			for (int c = 1 ; c < image.cols - 1; c++) {

				*output++ = saturate_cast<uchar>(5*current[c] - current[c-1] -
									current[c+1] - previous[c] - next[c]);
			}
		}
	}

	if (image.channels() == 3) {
		for (int r = 1; r < image.rows - 1; r++) {
			const uchar * previous = image.ptr<const uchar>(r - 1);
			const uchar * current = image.ptr<const uchar>(r);
			const uchar * next = image.ptr<const uchar>(r + 1);

			uchar *output = result.ptr<uchar>(r);

			for (int c = 3 ; c < 3*image.cols - 3; c++) {

				*output++ = saturate_cast<uchar>(5*current[c] - current[c-3] -
									current[c+3] - previous[c] - next[c]);
			}
		}
	}

	return result;
}

// sharpen3 - sharpens an image by taking a weighted average of each pixels with its neighbors
// preconditions: image is an greyscale or color byte image
// postconditions: a new image with the same size is returned after sharpening
Mat sharpen3(const Mat &image) {
	Mat result = image.clone();

	if (image.channels() == 1) {
		Mat_<uchar>::const_iterator in = image.begin<uchar>();
		Mat_<uchar>::iterator out = result.begin<uchar>();
		
		in += image.cols;
		result += image.cols;

		for (int r = 1; r < image.rows - 1; r++) {
			in++;
			out++;
			for (int c = 1 ; c < image.cols - 1; c++) {
				*out++ = saturate_cast<uchar>(5 * *in - *(in-1) - *(in+1) -
									 *(in + image.cols) - *(in - image.cols));
				in++;
			}
			in++;
			out++;
		}
	}

	if (image.channels() == 3) {
		Mat_<Vec3b>::const_iterator in = image.begin<Vec3b>();
		Mat_<Vec3b>::iterator out = result.begin<Vec3b>();
		
		in += image.cols;
		out += image.cols;

		for (int r = 1; r < image.rows - 1; r++) {
			in++;
			out++;
			for (int c = 1 ; c < image.cols - 1; c++) {
				for (int b = 0; b < 3; b++) {
					(*out)[b] = saturate_cast<uchar>(5 * (*in)[b] - (*(in-1))[b] - (*(in+1))[b] -
										 (*(in + image.cols))[b] - (*(in - image.cols))[b]);
				}
				in++;
				out++;
			}
			in++;
			out++;
		}
	}

	return result;
}


// main - exercises the sharpening methods
// precondition: boomer.jpg exists in the code directory and is a valid JPG
// postconditions: The original and sharpened images are displayed on the screen
//					Also creates an image photonegative and displays it on the screen
//					Note: waits for a key press between each image display
//					output.jpg (the sharpened image) is written to the disk
int main(int argc, char* argv[])
{	
		Mat image = imread("test3.jpg");	
		namedWindow("Original Image");
		imshow("Original Image", image);
		waitKey(0);

		Mat sharpened = sharpen1(image);
		namedWindow("Sharpened Image 1");
		imshow("Sharpened Image 1", sharpened);
		waitKey(0);

		sharpened = sharpen2(image);
		namedWindow("Sharpened Image 2");
		imshow("Sharpened Image 2", sharpened);
		waitKey(0);

		sharpened = sharpen3(image);
		namedWindow("Sharpened Image 3");
		imshow("Sharpened Image 3", sharpened);
		waitKey(0);

		imwrite("output.jpg", sharpened);

		for (int r = 0; r < image.rows; r++)
			for (int c = 0; c < image.cols; c++)
				for (int b = 0; b < 3; b++)
					image.at<Vec3b>(r,c)[b] = 255 - image.at<Vec3b>(r,c)[b];

		namedWindow("Photonegative");
		imshow("Photonegative", image);
		waitKey(0);

		return 0;
}