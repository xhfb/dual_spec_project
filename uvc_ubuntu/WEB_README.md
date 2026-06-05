# Web 测温测试程序（鼠标点选像素温度）

## 1. 编译底层采集程序

在 `uvc_ubuntu/` 目录：

```bash
make
```

## 2. 安装 Python 依赖

建议用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_web.txt
```

若启动时出现 `No supported WebSocket library detected`，说明 WebSocket 依赖未装好，请重新执行：

```bash
pip install -r requirements_web.txt
```

## 3. 启动 Web 服务

```bash
source .venv/bin/activate
uvicorn web_temp_viewer:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://<你的IP>:8000/`。

## 4. 使用说明

- 页面左侧实时显示画面（从 `./uvc_demo web` 读取）。
- 右侧会**始终显示画面中心点温度**。
- 用鼠标点击画面任意位置，右侧会额外显示该像素的温度（°C）。

## 4.1 画面左右拼接/错位的修正（可选）

如果发现画面出现“左右拼接缝”，通常是 YUYV 与温度矩阵存在行列偏移。可以用环境变量对两者同时做 `roll` 对齐：

```bash
export THERMAL_ROLL_X=-92   # 示例：水平方向像素滚动（正负都可试）
export THERMAL_ROLL_Y=0
uvicorn web_temp_viewer:app --host 0.0.0.0 --port 8000
```

## 5. 重要说明（温度换算）

目前服务端对温度 raw 单位做了**自动猜测**（常见 `K*100` 或 `C*10` 等）。
如果你发现温度值明显不对，需要根据设备协议把 `web_temp_viewer.py` 里的 `raw_to_celsius()` 固化成正确公式（例如发射率/反射温度/环境温度等补偿是否已在相机端完成）。

## 6. 调试参数自动落盘（给其他程序读取）

本程序会把以下参数自动保存到 JSON 文件（默认路径）：

- 默认文件：`uvc_ubuntu/thermal_cam_debug.json`
- 可用环境变量覆盖：`THERMAL_DEBUG_CONFIG=/path/to/your.json`

保存内容包括：

- `align.roll_x / align.roll_y`：WASD/方向键调整的对齐参数
- `calibration.gain / calibration.offset`：温度线性标定参数
- `calibration.two_point`：两点标定输入（已知温度）与最后一次点击坐标

你也可以通过 HTTP 读取当前内存中的配置：

- `GET /config`

