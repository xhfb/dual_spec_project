安装依赖
sudo apt update
sudo apt install build-essential libusb-1.0-0-dev libjpeg-dev cmake git

安装libuvc
git clone https://github.com/libuvc/libuvc.git
cd libuvc
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig

编译
make

运行
python img.py
由于图像可能存在偏移，按WASD进行修改图像偏移量。然后在py文件内对应修改。偏移量在成像的时候我已经打印好了