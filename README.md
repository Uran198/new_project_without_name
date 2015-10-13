Unnamed project
----

# TODO:
- make data to train and test on(different camera positions and different lightning and diffiring poses

# What is it?
TODO:

### Would be nice to try some of these:
* face detection
* clustering objects and recognition
* face recognition


# Idea 1
    Break an image on small objects. Naming those objects also would be nice

# Idea 2
    Capture images from web camera and do something with them. :) Like figure out am I happy or not today.

# Idea 3
    Presence detection to light on/off the screen

# Idea 4
    Polygon face

# Idea 5
    3d model construction

# Instaling OpenCV2 with Anaconda

Not sure that `BUILD_opencv_java` option is necessary, but it's how it worked for me. Replace `<HOME>` with path where you have anaconda3 installed.

	cmake \
	-D BUILD_opencv_java=OFF\
	-D CMAKE_BUILD_TYPE=Release \
	-D PYTHON3_EXECUTABLE=<HOME>/anaconda3/bin/python3.4m \
	-D PYTHON_INCLUDE_DIR=<HOME>/anaconda3/include/python3.4m/ \
	-D PYTHON_INCLUDE_DIR2=<HOME>/anaconda3/include/python3.4m/ \
	-D PYTHON_LIBRARY=<HOME>/anaconda3/lib/libpython3.4m.so \
	-D PYTHON3_PACKAGES_PATH=<HOME>/anaconda3/lib/python3.4/site-packages \
	-D PYTHON3_NUMPY_INCLUDE_DIRS=<HOME>/anaconda3/lib/python3.4/site-packages/numpy/core/include/ ..
	
