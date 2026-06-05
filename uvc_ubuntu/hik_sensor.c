#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "uvc_camera.h"
#include "libusb.h"

extern void  msleep(unsigned int ms);

///////////////////////////////////////////////////////
/* bmRequestType */
#define REQ_TYPE_SET      0x21
#define REQ_TYPE_GET      0xa1

/* bmRequest */
#define REQ_SET_CUR       0x01
#define REQ_GET_CUR       0x81
#define REQ_GET_LEN       0x85

/* wValue: CS ID */
#define XU_CS_ID_SYSTEM           (unsigned short)0x0100
/* sub-function ID */
#define SYSTEM_DEVICE_INFO        (unsigned char)0x01
#define SYSTEM_REBOOT             (unsigned char)0x02
#define SYSTEM_RESET              (unsigned char)0x03
#define SYSTEM_HARDWARE_SERVER    (unsigned char)0x04
#define SYSTEM_LOCALTIME          (unsigned char)0x05
#define SYSTEM_UPDATE_FIRMWARE    (unsigned char)0x06
#define SYSTEM_DIAGNOSED_DATA     (unsigned char)0x07
#define SYSTEM_UPDATE_STATE       (unsigned char)0x08

#define XU_CS_ID_IMAGE            (unsigned short)0x0200
#define IMAGE_BRIGHTNESS          (unsigned char)0x01
#define IMAGE_CONTRAST            (unsigned char)0x02
#define IMAGE_BACKGROUND_CORRECT  (unsigned char)0x03
#define IMAGE_MANUAL_CORRECT      (unsigned char)0x04
#define IMAGE_ENHANCEMENT         (unsigned char)0x05
#define IMAGE_VIDEO_ADJUST        (unsigned char)0x06

#define XU_CS_ID_THERMAL          (unsigned short)0x0300
#define THERMAL_BASIC_PARAM       (unsigned char)0x01
#define THERMAL_MODE              (unsigned char)0x02
#define THERMAL_MODE_NORMAL       1
#define THERMAL_MODE_EXPERT       2

#define THERMAL_ALG_VERSION       (unsigned char)0x04
#define THERMAL_STREAM_PARAM      (unsigned char)0x05
#define STREAM_TYPE_YUV_ONLY      6
#define STREAM_TYPE_YUV_RAW       9
#define STREAM_TYPE_YUV_TEMP      8

#define THERMAL_CALIBRATION_FILE  (unsigned char)0x0e
#define THERMAL_EXPERT_REGIONS    (unsigned char)0x0f
#define THERMAL_EXPERT_CORRECTION_PARAM    (unsigned char)0x10
#define THERMAL_EXPERT_CORRECTION_START    (unsigned char)0x11
#define THERMAL_TEMP_RISE_CALIBRATION      (unsigned char)0x12

#define XU_CS_ID_PROTOCOL_VER     (unsigned short)0x0400
#define XU_CS_ID_COMMAND_SWITCH   (unsigned short)0x0500
#define XU_CS_ID_ERROR_CODE       (unsigned short)0x0600

typedef struct _hms {
	unsigned short  msec;
	unsigned char   sec;
	unsigned char   min;
	unsigned char   hour;
	unsigned char   mday;
	unsigned char   mon;
	unsigned short  year;
	unsigned char   en;
} __attribute__((packed)) hms_t;

static void get_localtime(hms_t *hms)
{
	struct timeval tv;
	struct tm      now;
	time_t         t;

	gettimeofday(&tv, NULL);
	t = tv.tv_sec;
	localtime_r(&t, &now);
	hms->msec = (unsigned short)(tv.tv_usec/1000);
	hms->sec  = now.tm_sec;
	hms->min  = now.tm_min;
	hms->hour = now.tm_hour;
	hms->mday = now.tm_mday;
	hms->mon  = now.tm_mon + 1;
	hms->year = now.tm_year + 1900;
	hms->en   = 0;
}

static inline void  dump_localtime(hms_t *hms)
{
	printf("\n%d-%02d-%02d %02d:%02d:%02d.%d\n\n",
		hms->year, hms->mon, hms->mday, hms->hour, hms->min, hms->sec, hms->msec);
}

static int  get_len(uvc_t *u, unsigned short wValue, int retries)
{
	int   err, c = 0;
	unsigned char data[2];

	while (c < retries) {
		err = uvc_control_xfer(u, REQ_TYPE_GET, REQ_GET_LEN, wValue, 0x0a00, data, 2, 0);
		if (err == 2)
			return *((unsigned short*)data);

		printf("--- get_len error: %d ---\n", err);
		++c;
		msleep(500);
	}

	return -1;
}

static int  get_cur(uvc_t *u, unsigned short wValue, unsigned char *data, unsigned short len, int retries)
{
	int  err, c = 0;

	while (c < retries) {
		err = uvc_control_xfer(u, REQ_TYPE_GET, REQ_GET_CUR, wValue, 0x0a00, data, len, 0);
		if (err == len)
			return len;

		printf("--- get_cur error: %d ---\n", err);
		++c;
		msleep(500);
	}

	return -1;
}

static int  get_errno(uvc_t *u)
{
	int  len;
	unsigned char data[4];

	len = get_len(u, XU_CS_ID_ERROR_CODE, 10);
	if (len < 0) return -1;
	if (get_cur(u, XU_CS_ID_ERROR_CODE, data, 1, 10) < 0)
		return -1;

	printf("errno = %d\n", data[0]);
	return (int)data[0];
}

static inline int  set_cur(uvc_t *u, unsigned short wValue, unsigned char *data, unsigned short len, int retries)
{
	int  err, c = 0;

	while (c < retries) {
		err = uvc_control_xfer(u, REQ_TYPE_SET, REQ_SET_CUR, wValue, 0x0a00, data, len, 0);
		if (err == len)
			return len;
		printf("--- set_cur error: %d ---\n", err);
		++c;
		msleep(500);
	}

	return -1;
}

static int  set_curr_func(uvc_t *u, unsigned short cs_id, unsigned char sub_id, int retries)
{
	int  len;
	unsigned char  func[2];

	while (1) {
		len = get_len(u, XU_CS_ID_COMMAND_SWITCH, retries);
		if (len == 2) break;
		printf("--- get switch cmd len error: %d. ---\n", len);
		msleep(500);
	}

	func[0] = (cs_id>>8)&0xff;
	func[1] = sub_id;
	if (set_cur(u, XU_CS_ID_COMMAND_SWITCH, func, 2, retries) < 0) {
		printf("--- set switch cmd error. ---\n");
		return -1;
	}

	return 0;
}

static int  set_curr_data(uvc_t *u, unsigned short cs_id, unsigned char *buf, int retries)
{
	int  len;

	len = get_len(u, cs_id, retries);
	if (len < 0) {
		printf("--- get_len error. ---\n");
		return -1;
	}
	printf("\n--- set_curr_data: len = %d ---\n\n", len);
	if (set_cur(u, cs_id, buf, len, retries) < 0) {
		printf("--- set_curr error. ---\n");
		return -1;
	}

	return len;
}

static int  get_curr_data(uvc_t *u, unsigned short cs_id, unsigned char *buf, int retries)
{
	int  len;

	len = get_len(u, cs_id, retries);
	if (len < 0) {
		printf("--- get_len error. ---\n");
		return -1;
	}

	printf("\n--- get_curr_data: len = %d ---\n\n", len);

	if (get_cur(u, cs_id, buf, len, retries) < 0) {
		printf("-- get_curr error. ---\n");
		return -1;
	}

	return len;
}

static int  get_protocol_version(uvc_t *u, int retries)
{
	int  len;
	unsigned char data[4];

	len = get_len(u, XU_CS_ID_PROTOCOL_VER, retries);
	if (len < 0) {
		printf("--- get version len error. ---\n");
		return -1;
	}
	printf("--- proto version len: %d ---\n", len);
	if (get_cur(u, XU_CS_ID_PROTOCOL_VER, data, len, retries) < 0) {
		printf("--- get version error. ---\n");
		return -1;
	}
	printf("\n--- proto version: %s ---\n\n", data);

	return 0;
}

static void dump_devinfo(unsigned char *ptr)
{
	printf("\nfirmwareVersion: %s\n", ptr);
	ptr += 64;
	printf("encoderVersion: %s\n", ptr);
	ptr += 64;
	printf("hardwareVersion: %s\n", ptr);
	ptr += 64;
	printf("deviceName: %s\n", ptr);
	ptr += 64;
	printf("protocolVersion: %s\n", ptr);
	ptr += 4;
	printf("serialNumber: %s\n\n", ptr);
}

static unsigned int get_fw_version(uint8_t *d)
{
	char *ptr = (char *)d;

	ptr = strstr(ptr, "BUILD");
	ptr += 6;
	return (unsigned int)strtol(ptr, NULL, 10);
}

static int  get_device_info(uvc_t *u, int retries)
{
	unsigned char  rbuf[1600];
	unsigned int   fw_ver;

	printf("\n--- starting get_device_info ----\n");
	if (set_curr_func(u, XU_CS_ID_SYSTEM, SYSTEM_DEVICE_INFO, retries) < 0) return -1;
	if (get_curr_data(u, XU_CS_ID_SYSTEM, rbuf, retries) < 0) return -1;
	dump_devinfo(rbuf);
	fw_ver = get_fw_version(rbuf);
	printf("fw_ver = %u\n", fw_ver);

//	if (set_curr_func(u, XU_CS_ID_THERMAL, THERMAL_ALG_VERSION) < 0) return -1;
//	if (get_curr_data(u, XU_CS_ID_THERMAL, rbuf) < 0) return -1;
	return 0;
}

static int  wait_cmd_done(uvc_t *u, int after, int repeat)
{
	int   r = 0;

	msleep(after);
	while ((r = get_errno(u)) == 1)
		msleep(repeat);

	printf("finally errno = %d\n", r);
	return r;
}

static int  calibrate_time(uvc_t *u, int retries)
{
	hms_t hms;
	unsigned char data[1600] = {0};

	printf("\n--- starting calibrate_time ----\n");
	get_localtime(&hms);
	dump_localtime(&hms);
	if (set_curr_func(u, XU_CS_ID_SYSTEM, SYSTEM_LOCALTIME, retries) < 0) return -1;
	wait_cmd_done(u, 15, 15);
	if (set_curr_data(u, XU_CS_ID_SYSTEM, (unsigned char*)&hms, retries) < 0) return -1;
	wait_cmd_done(u, 100, 100);
	msleep(500);
	if (get_curr_data(u, XU_CS_ID_SYSTEM, data, retries) < 0) return -1;
	dump_localtime((hms_t *)data);

	return 0;
}

#ifdef DEBUG
static void dump_hex(uint8_t *d, int n, int max_col)
{
	uint8_t *ptr = d;
	int  i;
	int  left = n;

	while (left > max_col) {
		for (i = 0; i < max_col; ++i)
			printf("%02x ", *ptr++);
		printf("\n");
		left -= max_col;
	}

	if (left > 0) {
		for (i = 0; i < left; ++i)
			printf("%02x ", *ptr++);
		printf("\n");
	}
}
#endif

static inline int  is_same_config(uint8_t *d, uint8_t v)
{
	return (d[0] == 1&&d[1] == 1&&d[6] == v&&d[31] == 1);
}

static int  set_range(uvc_t *u, uint8_t *d, uint8_t v, int retries)
{
	d[0] = 1;
	d[1] = 1;
	d[2] = 0;
	d[3] = 0;
	d[4] = 0;
	d[5] = 1;
	d[6] = v;  //1-(30~45)  2-(-20~150)  3-(0~400)
	d[7] = 0;
	d[31] = 1;
	if (set_curr_data(u, XU_CS_ID_THERMAL, d, retries) < 0) return -1;
	return wait_cmd_done(u, 1000, 1000);
}

static int  thermal_base_config(uvc_t *u, int retries)
{
	uint8_t data[1600] = {0};
	int   n;

	printf("\n\n--- starting thermal_base_config ----\n");
	if (set_curr_func(u, XU_CS_ID_THERMAL, THERMAL_BASIC_PARAM, retries) != 0) return -1;
	wait_cmd_done(u, 15, 15);

	n = get_curr_data(u, XU_CS_ID_THERMAL, data, retries);
	if (n < 0) return -1;
#ifdef DEBUG
	if (n > 0)
		dump_hex(data, n, 16);
#endif


	if (is_same_config(data, 3))
		return 0;

	while (set_range(u, data, 3, retries) != 0);

#ifdef DEBUG
	memset(data, 0, sizeof(data));
	n = get_curr_data(u, XU_CS_ID_THERMAL, data, retries);
	if (n > 0)
		dump_hex(data, n, 16);
#endif

	printf("--- thermal_base_config done ----\n\n");
	return 0;
}

static int  image_enhance_config(uvc_t *u, unsigned char v, int retries)
{
	unsigned char data[1600] = {0};

	printf("\n--- starting image_enhance_config ----\n");
	if (set_curr_func(u, XU_CS_ID_IMAGE, IMAGE_ENHANCEMENT, retries) != 0) return -1;
	wait_cmd_done(u, 15, 15);
	if (get_curr_data(u, XU_CS_ID_IMAGE, data, retries) < 0) return -1;

	if (data[5] != v) {
	#ifdef DEBUG
		int   n;
	#endif
		data[0] = 1;
		data[5] = v;
		if (set_curr_data(u, XU_CS_ID_IMAGE, data, retries) < 0)
			return -1;
		wait_cmd_done(u, 100, 100);
	#ifdef DEBUG
		n = get_curr_data(u, XU_CS_ID_IMAGE, data, retries);
		if (n > 0)
			dump_hex(data, n, 16);
	#endif
	} else {
		printf("---- psuedo color meets my demands and not config again! ------\n");
	}

	return 0;
}

static int  stream_type_config(uvc_t *u, unsigned char type, int retries)
{
	unsigned char data[1600] = {0};

	printf("\n--- starting stream_type_config ----\n");
	if (set_curr_func(u, XU_CS_ID_THERMAL, THERMAL_STREAM_PARAM, retries) != 0) return -1;
	wait_cmd_done(u, 15, 15);
	if (get_curr_data(u, XU_CS_ID_THERMAL, data, retries) < 0)return -1;
	printf("0: chnlid = %d, stream_type = %d\n", data[0], data[1]);

	if (data[1] != type) {
		data[1] = type;
		if (set_curr_data(u, XU_CS_ID_THERMAL, data, retries) < 0)
			return -1;

		wait_cmd_done(u, 100, 100);

		get_curr_data(u, XU_CS_ID_THERMAL, data, retries);
		printf("1: chnlid = %d, stream_type = %d\n", data[0], data[1]);
	} else {
		printf("---- stream_type meets my demands and not config again! ------\n");
	}

	return 0;
}

int  hik_sensor_init(uvc_t *u)
{
	get_protocol_version(u, 20);
	if (get_device_info(u, 30) != 0) {
		printf("\n--- get_device_info error ---\n");
		return -1;
	}

	if (calibrate_time(u, 20) != 0) {
		printf("\n--- calibrate_time error ---\n");
		return -1;
	}

	if (thermal_base_config(u, 40) != 0) {
		printf("--- thermal_base_config error ---\n");
		return -1;
	}

	/* set psuedo color: 10 */
	if (image_enhance_config(u, 10, 40) != 0) {
		printf("\n--- image_enhance_config error ---\n");
		return -1;
	}

	/* set stream type: temperature+yuv */
	if (stream_type_config(u, STREAM_TYPE_YUV_TEMP, 20) != 0) {
		printf("\n--- stream_type_config error ---\n");
		return -1;
	}

	return 0;
}


