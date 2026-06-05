#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <limits.h>

#include "uvc_camera.h"

typedef struct _condwait {
	pthread_mutex_t        mutx;
	pthread_cond_t         cond;
	volatile unsigned int  cwaiters;
	volatile int           cval;
} condwait_t;

#define VID       0x2bdf
#define PID       0x0102
#define VPSTR    "2bdf:0102"
#define find_hik_sensor(s)   (strstr(s, VPSTR) != NULL)

static pthread_t      hk_tid;
static volatile int   hik_run = 0;
int    width, height;
int    g_fps;
static uvc_t   g_uvc;

/* 0: stdout 只输出YUV（兼容 img.py）
 * 1: stdout 输出 TempRaw(16bit)*W*H + YUV（供 Web 端解析/点选测温）
 */
static volatile int g_output_mode = 0;
void hik_set_output_mode(int mode) { g_output_mode = mode; }

#define SAFE_EXIT()  do { \
	hk_tid = 0; \
	kill(getpid(), SIGTERM); \
	pthread_exit(NULL); \
} while (0)

extern int   hik_sensor_init(uvc_t *u);

void  msleep(unsigned int ms)
{
	struct timeval  tv;

	tv.tv_sec = ms/1000;
	tv.tv_usec = (ms%1000)*1000;
	select(0, NULL, NULL, NULL, &tv);
}

static int  my_system(const char *cmd, char *obuf, int size)
{
	FILE  *fp;

	memset(obuf, 0, size);
	fp = popen(cmd, "r");
	if (fp) {
		int   n;
		n = fread(obuf, 1, size, fp);
		if (n >= 0)
			obuf[n] = 0;
		pclose(fp);
		return n;
	}

	return -1;
}

static void  condwait_init(condwait_t *cw)
{
	pthread_condattr_t cndattr;
	memset(cw, 0, sizeof(condwait_t));
	pthread_mutex_init(&cw->mutx, NULL);
	pthread_condattr_init(&cndattr);
	pthread_condattr_setclock(&cndattr, CLOCK_MONOTONIC);
	pthread_cond_init(&cw->cond, &cndattr);
	pthread_condattr_destroy(&cndattr);
}

static int   condwait_wait(condwait_t *cw)
{
	int  v;

	pthread_mutex_lock(&cw->mutx);
	++cw->cwaiters;
	while (cw->cval == 0)
		pthread_cond_wait(&cw->cond, &cw->mutx);
	--cw->cwaiters;
	v = cw->cval;
	pthread_mutex_unlock(&cw->mutx);

	return v;
}

static int  condwait_signal(condwait_t *cw, int v)
{
	int  ov;

	pthread_mutex_lock(&cw->mutx);
	ov = cw->cval;
	if (cw->cval == 0) {
		cw->cval = v;
		if (cw->cwaiters > 0)
			pthread_cond_signal(&cw->cond);
	}
	pthread_mutex_unlock(&cw->mutx);

	return ov;
}

typedef enum {
	TEMP_UNIT_RAW = 0,
	TEMP_UNIT_CENTI_KELVIN, /* raw = Kelvin * 100 */
	TEMP_UNIT_DECI_CELSIUS, /* raw = Celsius * 10 */
	TEMP_UNIT_CENTI_CELSIUS /* raw = Celsius * 100 */
} temp_unit_t;

static temp_unit_t __attribute__((unused)) guess_temp_unit(const uint16_t *raw, int pixels)
{
	/* 经验判别：
	 * - cK: 常见 28000~45000 (约 7℃~177℃) 或更大
	 * - dC: 常见 200~4000 (20℃~400℃)
	 * - cC: 常见 2000~400000 (20℃~4000℃) 但一般不会这么大；若在 2000~50000 也可能是 cC（20℃~500℃）
	 */
	int sample = pixels < 256 ? pixels : 256;
	uint32_t sum = 0;
	for (int i = 0; i < sample; ++i) sum += raw[i];
	uint32_t avg = sum / (uint32_t)sample;

	if (avg > 15000 && avg < 80000) return TEMP_UNIT_CENTI_KELVIN;
	if (avg > 0 && avg < 8000) return TEMP_UNIT_DECI_CELSIUS;
	if (avg >= 8000 && avg < 200000) return TEMP_UNIT_CENTI_CELSIUS;
	return TEMP_UNIT_RAW;
}

static float __attribute__((unused)) raw_to_celsius(uint16_t v, temp_unit_t unit)
{
	switch (unit) {
	case TEMP_UNIT_CENTI_KELVIN: return ((float)v) / 100.0f - 273.15f;
	case TEMP_UNIT_DECI_CELSIUS: return ((float)v) / 10.0f;
	case TEMP_UNIT_CENTI_CELSIUS: return ((float)v) / 100.0f;
	default: return (float)v;
	}
}

/* 如需在 C 端打印温度统计，可在这里加回调/打印逻辑；Web 端示例在 Python 解析 raw 后计算。 */

/* 回调函数：将视频数据直接写入标准输出，传给 Python */
static void frame_cb(uvc_frame_t *frame, void *arg)
{
    uvc_t *u = (uvc_t *)arg;

    if (!hik_run) return;
    if (frame == NULL || frame->data_bytes != u->size)
        return;

    /* 帧结构：4640字节头 + 温度( u->len ) + YUV( u->len ) */
    unsigned char *base = (unsigned char *)frame->data;
    unsigned char *temp_ptr = base + 4640;
    unsigned char *yuv_ptr  = base + u->offset; /* = 4640 + u->len */

    if (g_output_mode == 1) {
        /* Web 模式：先温度 raw 再输出 YUV，便于 Python 一次 read 固定长度解析 */
        fwrite(temp_ptr, 1, u->len, stdout);
        fwrite(yuv_ptr, 1, u->len, stdout);
    } else {
        /* 默认：兼容 img.py，只输出 YUV */
        fwrite(yuv_ptr, 1, u->len, stdout);
    }
    
    /* 强制刷新，保证实时性 */
    fflush(stdout);
}

static int  hik_capture_open(uvc_t *u)
{
	char  qstr[512];
	unsigned int  i = 0;

	while (1) {
		my_system("lsusb", qstr, sizeof(qstr));
		if (find_hik_sensor(qstr)) {
			printf("hik sensor found!\n");
			break;
		}

		if (i < 10) {
			printf("lsusb: %s\n", qstr);
			printf("hik sensor not found!\n");
		}
		++i;
		msleep(1000);
	}

	if (uvc_camera_open(u, VID, PID) == 0) {
		printf("--- hik_sensor_open successfully ---\n");
		return 0;
	}
	return -1;
}

static int  hik_capture_init(uvc_t *u, int *w, int *h)
{
	if (hik_sensor_init(u) != 0) {
		printf("hik sensor init failed!\n");
		return -1;
	}

	uvc_set_video_mode(u, w, h, &g_fps);

	width = *w;
	height = *h;

	return 0;
}

static int  hik_capture_start(uvc_t *u)
{
	uvc_set_frame_callback(u, frame_cb);
	if (uvc_camera_start(u) != 0) {
		uvc_camera_close(u);
		printf("uvc_camera_start failed\n");
		return -1;
	}

	return 0;
}

static void  hik_capture_close(uvc_t *u)
{
	uvc_camera_close_stream(u);
	uvc_camera_close(u);
}

static condwait_t  h_cw;
static void  *hik_capture_loop(void *arg)
{
	uvc_t *u = (uvc_t *)arg;

	printf("hik sensor: w = %d, h = %d\n", width, height);
	u->fptr = malloc(width*height*2);

	condwait_wait(&h_cw);
	if (hik_capture_start(u) != 0)
		SAFE_EXIT();
	hik_run = 1;
	while (hik_run) {
		pause();
	}
	hik_capture_close(u);
	printf("--- hik_capture_loop quit! ---\n");

	free(u->fptr);
	u->fptr = NULL;

	pthread_exit(NULL);
}

void  prepare_hik_sensor(void)
{
	int  w, h;

	condwait_init(&h_cw);
	if (hik_capture_open(&g_uvc) != 0) {
		printf("hik_capture_open error and exit!\n");
		exit(EXIT_FAILURE);
	}

	if (hik_capture_init(&g_uvc, &w, &h) != 0) {
		printf("hik_capture_init error and exit!\n");
		exit(EXIT_FAILURE);
	}

	pthread_create(&hk_tid, NULL, hik_capture_loop, (void *)&g_uvc);
}

void  start_hik_sensor(void)
{
	condwait_signal(&h_cw, 1);
}

void  stop_hik_sensor(void)
{
	hik_run = 0;
	if (hk_tid > 0) {
		pthread_kill(hk_tid, SIGINT);
		pthread_join(hk_tid, NULL);
		printf("hik sensor loop exit!\n");
	}
}


