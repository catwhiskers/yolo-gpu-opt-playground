sudo apt-get -y update
sudo apt-get -y upgrade
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.2.0 opencv
mv opencv_contrib-4.2.0 opencv_contrib
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
sudo bash -c 'echo "/usr/local/cuda-11.0/targets/x86_64-linux/lib/" > /etc/ld.so.conf.d/cuda.conf'
sudo ldconfig -v
cd opencv/cmake/;
cp ../../OpenCVDetectCUDA11-compatible.cmake OpenCVDetectCUDA.cmake
#wget -O OpenCVDetectCUDA.cmake https://raw.githubusercontent.com/catwhiskers/yolo-gpu-opt-playground/master/opencv-inf/OpenCVDetectCUDA11-compatible.cmake
cd ../;mkdir build; cd build/
sudo apt-get install -y build-essential cmake unzip pkg-config  libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev  libatlas-base-dev gfortran python3-dev wget unzip
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=7.5 -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=/usr/bin/python3 -D CMAKE_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/stubs/ -D BUILD_opencv_world=OFF -D BUILD_EXAMPLES=ON ..
make -j8
sudo make install
pip3.6 install imutils

