# vehicle-detect
Vehicle recognition with computer vision and machine learning. 
Images of all formats except .GIF can be fed into the program, and
Vehicle Detect will draw bounding boxes around what it thinks are
vehicles in the scene.

This is accomplished using fairly involved techniques including
Histogram of Oriented Gradients and Support Vector Machine. Great
detail is covered in the included paper and slides.

# Prerequisites
This project is implemented in C++ using OpenCV using Visual Studio 17.

Please download and install OpenCV to a convenient location:
https://sourceforge.net/projects/opencvlibrary/

If you want to compile or modify the code, please download and install
Microsoft Visual Studio 2017 Community (free):
https://visualstudio.microsoft.com/downloads/

# Ready to run
Navigate to:

	/VehicleDetect/VehicleDetect/test.bat

You can run this script to test the provided (test.png) image. If you
would like to supply your own image, download it to this directory
and edit the script to include the new filename.

# Compile
Using Visual Studio, open the project file at:

	/VehicleDetect/VehicleDetect.sln

Then click Debug > Run without Debugging... or simply press Ctrl+F5.

Note that when compiling, the test will run using the default test
image provided (test.png).