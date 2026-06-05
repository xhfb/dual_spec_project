import subprocess
import cv2
import numpy as np
import sys
import os

# --- 配置参数 ---
WIDTH = 256
HEIGHT = 192
FRAME_SIZE = WIDTH * HEIGHT * 2  # YUYV 每个像素 2 字节

def main():
    if not os.path.exists("./uvc_demo"):
        print("错误: 找不到 ./uvc_demo，请先执行 make 编译！")
        return

    # 启动 C 程序
    cmd = ["sudo", "./uvc_demo"]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr)
    except Exception as e:
        print(f"启动失败: {e}")
        return

    print(f"\n>>> 窗口已打开 ({WIDTH}x{HEIGHT})")
    print(">>> 【调试操作说明】:")
    print("    [A / D] : 左右微调 (水平)")
    print("    [W / S] : 上下微调 (垂直)")
    print("    [ESC/q] : 退出")

    # --- 【关键修改】在这里直接填入你测出的偏移量 ---
    shift_x = 92  # 水平像素偏移 (你测出的值)
    shift_y = -3  # 垂直行偏移 (你测出的值)
    # -------------------------------------------

    while True:
        raw_data = process.stdout.read(FRAME_SIZE)
        if not raw_data or len(raw_data) != FRAME_SIZE:
            break

        yuv = np.frombuffer(raw_data, dtype=np.uint8)

        try:
            yuv = yuv.reshape((HEIGHT, WIDTH, 2))
            
            # --- 核心调试逻辑 ---
            # 1. 垂直滚动 (axis=0)
            if shift_y != 0:
                yuv = np.roll(yuv, shift_y, axis=0)
            # 2. 水平滚动 (axis=1)
            if shift_x != 0:
                yuv = np.roll(yuv, shift_x, axis=1)

            # 计算对应的 C 代码字节修正值
            # 1个水平像素 = 2字节, 1行 = WIDTH * 2 字节
            total_byte_fix = (shift_y * WIDTH * 2) + (shift_x * 2)

            # 转为 BGR 显示
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
            display_img = cv2.resize(bgr, (WIDTH * 3, HEIGHT * 3), interpolation=cv2.INTER_NEAREST)
            
            # 在屏幕上打印调试信息
            info_text = f"X: {shift_x} | Y: {shift_y}"
            byte_text = f"Byte Offset Fix: {total_byte_fix}" # 应该会显示 -1352
            
            cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_img, byte_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_img, "Press W/S/A/D to adjust", (10, HEIGHT*3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Calibration Mode", display_img)
        except Exception as e:
            pass

        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('a'): shift_x -= 1
        elif key == ord('d'): shift_x += 1
        elif key == ord('w'): shift_y -= 1
        elif key == ord('s'): shift_y += 1

    process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()