# 使用本机 GCC
CC = gcc
# 移除嵌入式特有的 Strip 工具
STRIP = strip

# 库链接：链接 libusb-1.0, libuvc 和 pthread
LIBS = -lusb-1.0 -luvc -lpthread

# 编译选项：
# 1. -I/usr/local/include : 找已安装的 libuvc 公共头文件
# 2. -I/usr/include/libusb-1.0 : 找 libusb 头文件
# 3. -I./libuvc/include : 【新增】找 libuvc 源码里的私有头文件
# 4. -I./libuvc/build/include : 【新增】找编译生成的配置头文件
CFLAGS = -Wall -O2 -DDEBUG -I/usr/local/include -I/usr/include/libusb-1.0 -I./libuvc/include -I./libuvc/build/include

# 目标文件名
TARGET = uvc_demo

# 源文件
SOURCES = main.c hik_sensor.c hik_capture.c uvc_camera.c
OBJS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)
	# $(STRIP) $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)