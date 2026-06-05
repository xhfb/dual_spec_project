#include <string.h>

#include "libuvc/libuvc.h"
#include "libuvc/libuvc_internal.h"
#include "libusb.h"
#include "uvc_camera.h"

int  uvc_camera_open(uvc_t *u, int vid, int pid)
{
	int ret = 0;

	memset(u, 0, sizeof(uvc_t));
	ret = uvc_init(&u->ctx, NULL);
	if (ret < 0) return -1;

	ret = uvc_find_device(u->ctx, &u->device, vid, pid, NULL);
	if (ret < 0) {
		uvc_exit(u->ctx);
		memset(u, 0, sizeof(uvc_t));
		return -2;
	}

	ret = uvc_open(u->device, &u->devh);
	if (ret < 0) {
		uvc_unref_device(u->device);
		uvc_exit(u->ctx);
		memset(u, 0, sizeof(uvc_t));
		return -3;
	}

	return 0;
}

void uvc_camera_close(uvc_t *u)
{
	uvc_close(u->devh);
	uvc_unref_device(u->device);
	uvc_exit(u->ctx);
}

/*
	         256	              384	                640
set_resolution   256*400	      384*590	                640*1033
real_resolution  256*192              384*288                   640*512
temp length      256*192*2=98304      384*288*2=221184	        640*512*2=655360
yuv length       256*192*2=98304      384*288*2=221184	        640*512*2=655360
frame length     201248=4640+98304*2  447008=4640+221184*2	1315360=4640+655360*2
temp offset      4636+4=4640	      4636+4=4640	        4636+4=4640
yuv offset       4636+4+98304=102944  4636+4+221184=225824	4636+4+655360=660000
*/
void  uvc_set_video_mode(uvc_t *u, int *w, int *h, int *fps)
{
	int   width, height = 0;

	const uvc_format_desc_t *fmt_desc = uvc_get_format_descs(u->devh);
	const uvc_frame_desc_t  *frame_desc = fmt_desc->frame_descs;

	uvc_print_diag(u->devh, stderr);

	*w = frame_desc->wWidth;
	width = frame_desc->wWidth;
	*fps = 10000000/frame_desc->dwDefaultFrameInterval;

	if (width == 640) {
		*h = 512;
		width = 80;
		height = 8221;
		u->len = 640*512*2;
	} else if (width == 384) {
		*h = 288;
		height = 590;
		u->len = 384*288*2;
	} else if (width == 256) {
		*h = 192;
		height = 400;
		u->len = 256*192*2;
	}
	u->size = 4640+2*u->len;
	u->offset = u->len + 4640;
	/* 核心修改：应用 -1352 的字节偏移修正 */
    /* 原计算值: 102944, 修正后: 101592 */
    // u->offset = u->len + 4640 - 1352;

	printf("width=%d,height=%d,size=%d,len=%d,offset=%d\n", *w, *h, u->size, u->len, u->offset);

	uvc_get_stream_ctrl_format_size(u->devh, &u->ctrl, UVC_FRAME_FORMAT_YUYV,
	                                width, height, *fps);
}

int  uvc_camera_start(uvc_t *u)
{
	uvc_error_t err;
	err = uvc_stream_open_ctrl(u->devh, &u->stream, &u->ctrl);
	if (err != UVC_SUCCESS) {
		uvc_perror(err, "open_ctrl");
		printf("open_ctrl error!\n");
		return -1;
	}
	err = uvc_stream_start(u->stream, u->frame_cb, u, 0);
	if (err != UVC_SUCCESS) {
		uvc_perror(err, "stream_start");
		uvc_stream_close(u->stream);
		printf("stream_start error!\n");
		return -1;
	}

	return 0;
}

void  uvc_camera_close_stream(uvc_t *u)
{
	if (u->stream) {
		uvc_stream_close(u->stream);
		u->stream = NULL;
	}
}

int  uvc_camera_get_frame(uvc_t *u, unsigned int timeout_us)
{
	uvc_error_t err;

	u->frame = NULL;
	err = uvc_stream_get_frame(u->stream, &u->frame, timeout_us);
	return (int)err;
}

int  uvc_control_xfer(uvc_t *u, unsigned char bmRequestType, unsigned char bRequest,
                             unsigned short wValue, unsigned short wIndex,
	                     unsigned char *data, uint16_t wLength, unsigned int timeout)
{
	return libusb_control_transfer(u->devh->usb_devh, bmRequestType, bRequest, wValue, wIndex,
		                       data, wLength, timeout);
}


