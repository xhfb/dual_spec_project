#ifndef __UVC_CAMERA_H__
#define __UVC_CAMERA_H__

#include "libuvc/libuvc.h"

#define FORMAT_ANY               "any"
#define FORMAT_UNCOMPRESSED      "uncompressed"
#define FORMAT_COMPRESSED        "compressed"
#define FORMAT_YUYV              "yuyv"
#define FORMAT_UYVY              "uyvy"
#define FORMAT_RGB               "rgb"
#define FORMAT_BGR               "bgr"
#define FORMAT_MJPEG             "mjpeg"
#define FORMAT_GRAY8             "gray8"

typedef struct _uvc {
	uvc_context_t        *ctx;
	uvc_device_t         *device;
	uvc_device_handle_t  *devh;
	uvc_stream_ctrl_t     ctrl;
	uvc_stream_handle_t  *stream;
	uvc_frame_t          *frame;
	uvc_frame_callback_t *frame_cb; //(uvc_frame_t* frame, void* ptr)
	int                   size;
	int                   offset;
	int                   len;
	unsigned char        *fptr;
} uvc_t;

#define uvc_set_frame_callback(u, cb)  (u)->frame_cb = (cb)

int   uvc_camera_open(uvc_t *u, int vid, int pid);
void  uvc_camera_close(uvc_t *u);
void  uvc_set_video_mode(uvc_t *u, int *w, int *h, int *fps);
int   uvc_camera_start(uvc_t *u);
void  uvc_camera_close_stream(uvc_t *u);
int   uvc_camera_get_frame(uvc_t *u, unsigned int timeout_us);
int   uvc_control_xfer(uvc_t *u, unsigned char bmRequestType, unsigned char bRequest,
                             unsigned short wValue, unsigned short wIndex,
	                     unsigned char *data, uint16_t wLength, unsigned int timeout);


#endif

