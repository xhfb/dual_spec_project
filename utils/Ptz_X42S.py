"""云台控制模块 - 基于ZDT_X42S第二代闭环步进电机(Emm固件).

使用emm_stepper库控制二自由度云台，由两个ZDT_X42S电机驱动。
仅包含yaw(水平)和pitch(俯仰)两轴，不含motor3。
"""

import serial
from serial import Serial
from emm_stepper import EmmDevice, Direction, MotionMode
import time
import logging

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Gimbal:
    """二自由度云台类.
    
    使用两个ZDT_X42S电机驱动的云台，支持yaw(水平)和pitch(俯仰)两个轴的控制。
    
    使用示例:
        ```python
        # 创建云台实例
        ptz = Gimbal(
            serial_port="COM50",
            yaw_id=1,
            pitch_id=2,
            yaw_ratio=8,
            pitch_ratio=4
        )
        
        # 设置速度模式运动
        ptz.set_speed(pitch_speed=10, yaw_speed=20)
        
        # 设置位置模式运动(角度)
        ptz.set_position(pitch_angle=30, yaw_angle=45)
        
        # 关闭云台
        del ptz
        ```
    """

    def __init__(
        self,
        serial_port: str,
        yaw_id: int = 1,
        pitch_id: int = 2,
        yaw_ratio: float = 8.0,
        pitch_ratio: float = 4.0,
        baudrate: int = 115200,
        default_speed: int = 300,
        default_acceleration: int = 50,
        microstep: int = 16,
        yaw_limits: tuple = None,
        pitch_limits: tuple = None,
    ):
        """初始化云台.
        
        Args:
            serial_port: 串口号 (如 "COM50" 或 "/dev/ttyUSB0")
            yaw_id: yaw轴电机ID (1-255)
            pitch_id: pitch轴电机ID (1-255)
            yaw_ratio: yaw轴减速比
            pitch_ratio: pitch轴减速比
            baudrate: 串口波特率 (默认115200)
            default_speed: 默认运动速度 (RPM, 0-3000)
            default_acceleration: 默认加速度档位 (0-255)
            microstep: 电机细分值 (默认16)
            yaw_limits: yaw轴角度限位 (min_deg, max_deg)，None表示不限位
            pitch_limits: pitch轴角度限位 (min_deg, max_deg)，None表示不限位
        """
        # 保存参数
        self.yaw_ratio = yaw_ratio
        self.pitch_ratio = pitch_ratio
        self.microstep = microstep
        self.default_speed = default_speed
        self.default_acceleration = default_acceleration
        self.yaw_limits = yaw_limits
        self.pitch_limits = pitch_limits

        # 计算每圈脉冲数 (1.8°电机，一圈200步)
        self.pulses_per_revolution = 200 * microstep
        
        # 创建串口连接
        self.ser = serial.Serial(serial_port, baudrate, timeout=0.1)
        logging.info(f"初始化云台串口 {serial_port}")
        
        # 创建电机设备实例
        self.motor_yaw = EmmDevice(
            serial_connection=self.ser,
            address=yaw_id,
            auto_test=True
        )
        self.motor_pitch = EmmDevice(
            serial_connection=self.ser,
            address=pitch_id,
            auto_test=True
        )
        logging.info(f"初始化云台电机 (yaw_id={yaw_id}, pitch_id={pitch_id})")
        
        # 清除保护状态
        self.motor_yaw.clear_protection()
        self.motor_pitch.clear_protection()
        logging.info("清除电机保护状态")
        
        # 使能电机
        self.motor_yaw.enable()
        self.motor_pitch.enable()
        logging.info("使能云台电机")
        
        # 记录上次命令，用于避免重复发送
        self._last_speed_cmd = (0, 0)
        self._last_position_cmd = (0.0, 0.0)

    def _angle_to_pulses(self, angle: float, ratio: float) -> int:
        """将角度转换为脉冲数.
        
        Args:
            angle: 目标角度 (度)
            ratio: 减速比
            
        Returns:
            脉冲数
        """
        # 考虑减速比后的脉冲数
        # 电机转 ratio 圈，输出轴转 1 圈
        pulses = int(angle / 360.0 * self.pulses_per_revolution * ratio)
        return pulses

    def enable_all(self) -> None:
        """使能所有电机."""
        self.motor_yaw.enable()
        self.motor_pitch.enable()
        logging.info("云台使能")

    def disable_all(self) -> None:
        """失能所有电机."""
        self.motor_yaw.disable()
        self.motor_pitch.disable()
        logging.info("云台失能")

    def stop_all(self) -> None:
        """停止所有电机运动."""
        self.motor_yaw.stop()
        self.motor_pitch.stop()
        logging.info("云台停止")

    def clear_protection(self) -> None:
        """清除所有电机的保护状态."""
        self.motor_yaw.clear_protection()
        self.motor_pitch.clear_protection()
        logging.info("清除保护状态")

    def calibrate_pitch(
        self,
        calibrate_speed: int = 30,
        calibrate_direction: Direction = Direction.CW,
        back_angle: float = -45.0,
        timeout: float = 30.0,
    ) -> bool:
        """校准pitch轴 - 通过堵转检测自动回中.
        
        pitch轴电机向指定方向旋转，当检测到堵转后，向反方向转动特定角度回到水平位置，
        然后将该位置设为零点。
        
        Args:
            calibrate_speed: 校准时的运动速度 (RPM, 建议使用较低速度)
            calibrate_direction: 校准时的运动方向 (向限位方向运动)
            back_angle: 堵转后回退的角度 (度, 由云台机械结构决定, 负值表示反向)
            timeout: 超时时间 (秒)
            
        Returns:
            是否校准成功
        """
        logging.info("开始校准pitch轴...")

        # 校准时降低堵转检测电流（0.5A），避免校准过程电流过大
        # 使用 emm_stepper 的 get_config()/set_config()，并在结束后恢复原值。
        prev_stall_current = None
        try:
            cfg = self.motor_pitch.get_config()
            if hasattr(cfg, "stall_current"):
                prev_stall_current = int(cfg.stall_current)
                cfg.stall_current = 500  # 0.5A = 500mA
                try:
                    self.motor_pitch.set_config(cfg, store=False)
                except TypeError:
                    # 兼容旧版本接口（若无store参数则直接调用）
                    self.motor_pitch.set_config(cfg)
                logging.info("pitch轴校准：堵转检测电流设为 500mA (0.5A)")
        except Exception as e:
            logging.warning(f"pitch轴校准：设置堵转检测电流失败: {e}")

        try:
            # 确保电机使能
            self.motor_pitch.enable()
            time.sleep(0.1)

            # 停止当前运动
            self.motor_pitch.stop()
            time.sleep(0.1)

            # 清除保护状态
            self.motor_pitch.clear_protection()
            time.sleep(0.1)

            # 开始向限位方向运动
            logging.info(f"pitch轴开始向 {calibrate_direction} 方向运动，速度={calibrate_speed} RPM")
            self.motor_pitch.jog(
                speed=calibrate_speed,
                direction=calibrate_direction,
                acceleration=10
            )

            # 等待堵转
            start_time = time.time()
            stall_detected = False

            while time.time() - start_time < timeout:
                status = self.motor_pitch.get_motor_status()
                if status.stall_detected or status.stall_protected:
                    stall_detected = True
                    logging.info("pitch轴检测到堵转")
                    break
                time.sleep(0.05)

            # 停止运动
            self.motor_pitch.stop()
            time.sleep(0.1)

            if not stall_detected:
                logging.warning("pitch轴校准超时，未检测到堵转")
                return False

            # 清除堵转保护
            self.motor_pitch.clear_protection()
            time.sleep(0.1)

            # 将当前位置清零（堵转位置作为临时零点）
            self.motor_pitch.zero_position()
            time.sleep(0.1)

            # 回退到水平位置
            back_pulses = self._angle_to_pulses(back_angle, self.pitch_ratio)
            logging.info(f"pitch轴回退 {back_angle}° (脉冲数: {back_pulses})")

            self.motor_pitch.move_pulses(
                pulse_count=back_pulses,
                speed=self.default_speed,
                acceleration=self.default_acceleration,
                motion_mode=MotionMode.RELATIVE_LAST
            )

            # 等待到位
            wait_start = time.time()
            while time.time() - wait_start < 10.0:
                status = self.motor_pitch.get_motor_status()
                if status.position_reached:
                    break
                time.sleep(0.05)

            time.sleep(0.5)

            # 将当前位置设为零点（水平位置）
            self.motor_pitch.zero_position()
            self._last_position_cmd = (0.0, self._last_position_cmd[1])

            logging.info("pitch轴校准完成，当前位置已设为零点")
            return True
        finally:
            if prev_stall_current is not None:
                try:
                    cfg = self.motor_pitch.get_config()
                    if hasattr(cfg, "stall_current"):
                        cfg.stall_current = int(prev_stall_current)
                        try:
                            self.motor_pitch.set_config(cfg, store=False)
                        except TypeError:
                            self.motor_pitch.set_config(cfg)
                        logging.info(
                            "pitch轴校准：已恢复堵转检测电流为 {}mA".format(prev_stall_current)
                        )
                except Exception as e:
                    logging.warning(f"pitch轴校准：恢复堵转检测电流失败: {e}")


    def _clear_yaw_stall_state(self, settle_time: float = 0.3) -> bool:
        """停止yaw电机并等待堵转/保护标志清除."""
        self.motor_yaw.stop()
        time.sleep(0.1)
        self.motor_yaw.clear_protection()
        time.sleep(settle_time)

        deadline = time.time() + 3.0
        while time.time() < deadline:
            status = self.motor_yaw.get_motor_status()
            if not status.stall_detected and not status.stall_protected:
                return True
            self.motor_yaw.clear_protection()
            time.sleep(0.1)

        logging.warning("yaw轴堵转标志未在超时内完全清除")
        return False

    def calibrate_yaw(
        self,
        calibrate_speed: int = 30,
        first_direction: Direction = Direction.CW,
        timeout: float = 30.0,
        stall_interval: float = 1.1,
        second_stall_detect_delay: float = 2.0,
        second_stall_min_travel_motor_deg: float = 20.0,
    ) -> bool:
        """校准yaw轴 - 双边堵转检测自动回中.
        
        向第一个方向运动直至堵转并记录位置，再向反方向运动直至堵转并记录位置，
        两次堵转检测之间间隔大于 stall_interval 秒；第二次堵转在启动反向后须等待
        second_stall_detect_delay 秒才开始检测，且须离开第一侧堵转位置一定距离后才认定堵转，
        然后运动到两堵转位置的中点并清零角度。
        
        Args:
            calibrate_speed: 校准时的运动速度 (RPM, 建议使用较低速度)
            first_direction: 第一次校准运动方向
            timeout: 单次堵转检测超时时间 (秒)
            stall_interval: 第一次堵转结束后到启动反向运动前的等待 (秒, 须大于1s)
            second_stall_detect_delay: 反向jog启动后延迟多久才开始第二次堵转检测 (秒)
            second_stall_min_travel_motor_deg: 第二次堵转前相对第一侧位置的最小电机轴位移 (度)
            
        Returns:
            是否校准成功
        """
        if stall_interval <= 1.0:
            stall_interval = 1.1
            logging.warning("stall_interval 须大于1s，已调整为 1.1s")

        logging.info("开始校准yaw轴（双边堵转）...")

        prev_stall_current = None
        try:
            cfg = self.motor_yaw.get_config()
            if hasattr(cfg, "stall_current"):
                prev_stall_current = int(cfg.stall_current)
                cfg.stall_current = 500  # 0.5A = 500mA
                try:
                    self.motor_yaw.set_config(cfg, store=False)
                except TypeError:
                    self.motor_yaw.set_config(cfg)
                logging.info("yaw轴校准：堵转检测电流设为 500mA (0.5A)")
        except Exception as e:
            logging.warning(f"yaw轴校准：设置堵转检测电流失败: {e}")

        try:
            self.motor_yaw.enable()
            time.sleep(0.1)
            self._clear_yaw_stall_state()

            second_direction = (
                Direction.CCW if first_direction == Direction.CW else Direction.CW
            )

            def _wait_stall(
                direction: Direction,
                label: str,
                detect_delay: float = 0.0,
                ref_position: float | None = None,
                min_travel_motor_deg: float = 0.0,
            ) -> float | None:
                logging.info(
                    f"yaw轴开始向 {direction} 方向运动，速度={calibrate_speed} RPM ({label})"
                )
                if detect_delay > 0:
                    logging.info(
                        f"yaw轴 {label}：启动后 {detect_delay:.1f}s 内不检测堵转"
                    )
                self.motor_yaw.jog(
                    speed=calibrate_speed,
                    direction=direction,
                    acceleration=10,
                )
                start_time = time.time()
                stall_pos = None
                while time.time() - start_time < timeout:
                    elapsed = time.time() - start_time
                    if elapsed < detect_delay:
                        time.sleep(0.05)
                        continue

                    status = self.motor_yaw.get_motor_status()
                    if status.stall_detected or status.stall_protected:
                        current_pos = self.motor_yaw.get_realtime_position()
                        if ref_position is not None and min_travel_motor_deg > 0:
                            traveled = abs(current_pos - ref_position)
                            if traveled < min_travel_motor_deg:
                                logging.debug(
                                    f"yaw轴 {label}：堵转信号已出现但位移不足 "
                                    f"({traveled:.2f}° < {min_travel_motor_deg:.2f}°)，忽略"
                                )
                                time.sleep(0.05)
                                continue

                        stall_pos = current_pos
                        logging.info(
                            f"yaw轴 {label} 堵转，电机位置={stall_pos:.2f}° "
                            f"(运动后 {elapsed:.2f}s)"
                        )
                        break
                    time.sleep(0.05)

                self.motor_yaw.stop()
                time.sleep(0.1)
                if stall_pos is None:
                    logging.warning(f"yaw轴 {label} 校准超时，未检测到堵转")
                    return None

                self._clear_yaw_stall_state()
                return stall_pos

            pos_first = _wait_stall(first_direction, "第一侧")
            if pos_first is None:
                return False

            logging.info(
                f"yaw轴等待 {stall_interval:.1f}s 后清除堵转状态并启动反向运动..."
            )
            time.sleep(stall_interval)
            self._clear_yaw_stall_state(settle_time=0.5)

            pos_second = _wait_stall(
                second_direction,
                "第二侧",
                detect_delay=second_stall_detect_delay,
                ref_position=pos_first,
                min_travel_motor_deg=second_stall_min_travel_motor_deg,
            )
            if pos_second is None:
                return False

            middle_motor = (pos_first + pos_second) / 2.0
            current_motor = self.motor_yaw.get_realtime_position()
            motor_delta = middle_motor - current_motor
            output_delta = motor_delta / self.yaw_ratio
            move_pulses = self._angle_to_pulses(output_delta, self.yaw_ratio)

            logging.info(
                f"yaw轴堵转位置: 第一侧={pos_first:.2f}°, "
                f"第二侧={pos_second:.2f}°, 中点={middle_motor:.2f}°"
            )
            logging.info(
                f"yaw轴移动到中间位置，输出轴增量={output_delta:.2f}° "
                f"(脉冲数: {move_pulses})"
            )

            if move_pulses != 0:
                self.motor_yaw.move_pulses(
                    pulse_count=move_pulses,
                    speed=self.default_speed,
                    acceleration=self.default_acceleration,
                    motion_mode=MotionMode.RELATIVE_LAST,
                )
                wait_start = time.time()
                while time.time() - wait_start < 10.0:
                    status = self.motor_yaw.get_motor_status()
                    if status.position_reached:
                        break
                    time.sleep(0.05)
                time.sleep(0.5)

            self.motor_yaw.zero_position()
            self._last_position_cmd = (self._last_position_cmd[0], 0.0)
            logging.info("yaw轴双边堵转校准完成，中点位置已设为零点")
            return True
        finally:
            if prev_stall_current is not None:
                try:
                    cfg = self.motor_yaw.get_config()
                    if hasattr(cfg, "stall_current"):
                        cfg.stall_current = int(prev_stall_current)
                        try:
                            self.motor_yaw.set_config(cfg, store=False)
                        except TypeError:
                            self.motor_yaw.set_config(cfg)
                        logging.info(
                            "yaw轴校准：已恢复堵转检测电流为 {}mA".format(
                                prev_stall_current
                            )
                        )
                except Exception as e:
                    logging.warning(f"yaw轴校准：恢复堵转检测电流失败: {e}")

    def calibrate_all(
        self,
        pitch_calibrate_speed: int = 30,
        pitch_calibrate_direction: Direction = Direction.CW,
        pitch_back_angle: float = -45.0,
        yaw_calibrate_speed: int = 30,
        yaw_first_direction: Direction = Direction.CW,
        yaw_stall_interval: float = 1.1,
        yaw_second_stall_detect_delay: float = 2.0,
    ) -> bool:
        """校准所有轴.
        
        Args:
            pitch_calibrate_speed: pitch轴校准速度
            pitch_calibrate_direction: pitch轴校准方向
            pitch_back_angle: pitch轴回退角度
            yaw_calibrate_speed: yaw轴校准速度
            yaw_first_direction: yaw轴第一次堵转检测方向
            yaw_stall_interval: yaw轴第一次堵转后到反向运动前的等待 (秒, 须大于1s)
            yaw_second_stall_detect_delay: yaw轴反向运动后延迟第二次堵转检测 (秒)
            
        Returns:
            是否全部校准成功
        """
        pitch_ok = self.calibrate_pitch(
            calibrate_speed=pitch_calibrate_speed,
            calibrate_direction=pitch_calibrate_direction,
            back_angle=pitch_back_angle,
        )

        yaw_ok = self.calibrate_yaw(
            calibrate_speed=yaw_calibrate_speed,
            first_direction=yaw_first_direction,
            stall_interval=yaw_stall_interval,
            second_stall_detect_delay=yaw_second_stall_detect_delay,
        )

        if pitch_ok and yaw_ok:
            logging.info("云台全部轴校准完成")
        else:
            logging.warning(
                f"云台校准未全部成功 (pitch={pitch_ok}, yaw={yaw_ok})"
            )
        return pitch_ok and yaw_ok

    def zero_position(self) -> None:
        """将当前位置设为零点."""
        self.motor_yaw.zero_position()
        self.motor_pitch.zero_position()
        self._last_position_cmd = (0.0, 0.0)
        logging.info("云台位置清零")

    def set_speed(
        self,
        pitch_speed: int = 0,
        yaw_speed: int = 0,
        acceleration: int = None,
    ) -> None:
        """设置速度模式运动.
        
        电机以指定速度持续运动，直到收到新的速度命令或停止命令。
        速度为正表示正向旋转，速度为负表示反向旋转，速度为0表示停止。
        
        注意: 速度单位为RPM，是电机轴的转速，不是输出轴的转速。
        输出轴转速 = 电机轴转速 / 减速比
        
        Args:
            pitch_speed: pitch轴速度 (RPM, 正负表示方向, 0-3000)
            yaw_speed: yaw轴速度 (RPM, 正负表示方向, 0-3000)
            acceleration: 加速度档位 (0-255, None则使用默认值)
        """
        if acceleration is None:
            acceleration = self.default_acceleration
        
        last_pitch_speed, last_yaw_speed = self._last_speed_cmd
        
        # 只有速度变化时才发送命令
        if last_pitch_speed != pitch_speed:
            if pitch_speed == 0:
                self.motor_pitch.stop()
            else:
                direction = Direction.CW if pitch_speed > 0 else Direction.CCW
                self.motor_pitch.jog(
                    speed=abs(pitch_speed),
                    direction=direction,
                    acceleration=acceleration
                )
        
        if last_yaw_speed != yaw_speed:
            if yaw_speed == 0:
                self.motor_yaw.stop()
            else:
                direction = Direction.CW if yaw_speed > 0 else Direction.CCW
                self.motor_yaw.jog(
                    speed=abs(yaw_speed),
                    direction=direction,
                    acceleration=acceleration
                )
        
        self._last_speed_cmd = (pitch_speed, yaw_speed)

    def set_position(
        self,
        pitch_angle: float = None,
        yaw_angle: float = None,
        speed: int = None,
        acceleration: int = None,
        motion_mode: MotionMode = MotionMode.ABSOLUTE,
    ) -> None:
        """设置位置模式运动.
        
        电机运动到指定的角度位置。角度是相对于零点的绝对位置。
        
        Args:
            pitch_angle: pitch轴目标角度 (度, None表示不改变)
            yaw_angle: yaw轴目标角度 (度, None表示不改变)
            speed: 运动速度 (RPM, 0-3000, None则使用默认值)
            acceleration: 加速度档位 (0-255, None则使用默认值)
            motion_mode: 运动模式 (默认绝对位置模式)
        """
        if speed is None:
            speed = self.default_speed
        if acceleration is None:
            acceleration = self.default_acceleration
        
        # 角度限位 clamp
        if pitch_angle is not None and self.pitch_limits is not None:
            pitch_angle = max(self.pitch_limits[0], min(self.pitch_limits[1], pitch_angle))
        if yaw_angle is not None and self.yaw_limits is not None:
            yaw_angle = max(self.yaw_limits[0], min(self.yaw_limits[1], yaw_angle))
        
        last_pitch_angle, last_yaw_angle = self._last_position_cmd
        
        # 只有位置变化时才发送命令
        if pitch_angle is not None and last_pitch_angle != pitch_angle:
            pulse_count = self._angle_to_pulses(pitch_angle, self.pitch_ratio)
            self.motor_pitch.move_pulses(
                pulse_count=pulse_count,
                speed=speed,
                acceleration=acceleration,
                motion_mode=motion_mode
            )
            self._last_position_cmd = (pitch_angle, self._last_position_cmd[1])
        
        if yaw_angle is not None and last_yaw_angle != yaw_angle:
            pulse_count = self._angle_to_pulses(yaw_angle, self.yaw_ratio)
            self.motor_yaw.move_pulses(
                pulse_count=pulse_count,
                speed=speed,
                acceleration=acceleration,
                motion_mode=motion_mode
            )
            self._last_position_cmd = (self._last_position_cmd[0], yaw_angle)

    def move_relative(
        self,
        pitch_delta: float = 0,
        yaw_delta: float = 0,
        speed: int = None,
        acceleration: int = None,
    ) -> None:
        """相对位置运动.
        
        电机从当前位置运动指定的角度增量。
        
        Args:
            pitch_delta: pitch轴角度增量 (度)
            yaw_delta: yaw轴角度增量 (度)
            speed: 运动速度 (RPM, 0-3000, None则使用默认值)
            acceleration: 加速度档位 (0-255, None则使用默认值)
        """
        if speed is None:
            speed = self.default_speed
        if acceleration is None:
            acceleration = self.default_acceleration
        
        if pitch_delta != 0:
            pulse_count = self._angle_to_pulses(pitch_delta, self.pitch_ratio)
            self.motor_pitch.move_pulses(
                pulse_count=pulse_count,
                speed=speed,
                acceleration=acceleration,
                motion_mode=MotionMode.RELATIVE_LAST
            )
        
        if yaw_delta != 0:
            pulse_count = self._angle_to_pulses(yaw_delta, self.yaw_ratio)
            self.motor_yaw.move_pulses(
                pulse_count=pulse_count,
                speed=speed,
                acceleration=acceleration,
                motion_mode=MotionMode.RELATIVE_LAST
            )

    def get_position(self) -> tuple:
        """获取当前位置.
        
        Returns:
            (pitch_angle, yaw_angle) 当前角度位置 (度)
        """
        pitch_pos = self.motor_pitch.get_realtime_position()
        yaw_pos = self.motor_yaw.get_realtime_position()
        
        # 考虑减速比转换为输出轴角度
        pitch_angle = pitch_pos / self.pitch_ratio
        yaw_angle = yaw_pos / self.yaw_ratio
        
        return (pitch_angle, yaw_angle)

    def get_speed(self) -> tuple:
        """获取当前速度.
        
        Returns:
            (pitch_speed, yaw_speed) 当前速度 (RPM, 电机轴转速)
        """
        pitch_speed = self.motor_pitch.get_realtime_speed()
        yaw_speed = self.motor_yaw.get_realtime_speed()
        return (pitch_speed, yaw_speed)

    def is_position_reached(self) -> tuple:
        """检查是否到达目标位置.
        
        Returns:
            (pitch_reached, yaw_reached) 是否到位
        """
        pitch_status = self.motor_pitch.get_motor_status()
        yaw_status = self.motor_yaw.get_motor_status()
        return (pitch_status.position_reached, yaw_status.position_reached)

    def wait_for_position(self, timeout: float = 10.0) -> bool:
        """等待所有轴到达目标位置.
        
        Args:
            timeout: 超时时间 (秒)
            
        Returns:
            是否全部到位
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            pitch_reached, yaw_reached = self.is_position_reached()
            if pitch_reached and yaw_reached:
                return True
            time.sleep(0.05)
        return False

    def __del__(self):
        """析构函数，关闭云台."""
        try:
            # 停止所有运动
            self.motor_pitch.stop()
            self.motor_yaw.stop()
            # 关闭串口
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
            logging.info("关闭云台对象")
        except Exception as e:
            logging.warning(f"关闭云台时出错: {e}")


if __name__ == "__main__":
    # 测试代码
    ptz = Gimbal(
        serial_port="/dev/ttyS1",
        yaw_id=2,
        pitch_id=1,
        yaw_ratio=8,
        pitch_ratio=4
    )
    
    # 校准云台
    print("校准云台...")
    ptz.calibrate_pitch(
        calibrate_speed=30,
        calibrate_direction=Direction.CW,
        back_angle=-60  # 根据实际机械结构调整
    )
    ptz.calibrate_yaw(
        stall_interval=0,              # 第一次堵转后静止等待
        second_stall_detect_delay=1.5,   # 反向运动后再开始检测
        second_stall_min_travel_motor_deg=30.0,
    )
    ptz.zero_position()
    # 测试位置模式
    print("测试位置模式...")
    ptz.set_position(pitch_angle=10, yaw_angle=10)
    ptz.wait_for_position()
    
    
    ptz.set_position(pitch_angle=-10, yaw_angle=-10)
    ptz.wait_for_position()

    
    ptz.set_position(pitch_angle=0, yaw_angle=0)
    ptz.wait_for_position()
    
    # 测试速度模式
    print("测试速度模式...")
    ptz.set_speed(pitch_speed=5, yaw_speed=5)
    time.sleep(2)
    
    ptz.set_speed(pitch_speed=-5, yaw_speed=-5)
    time.sleep(2)
    
    ptz.set_speed(pitch_speed=0, yaw_speed=0)
    time.sleep(2)
    
    # # 获取当前位置
    pos = ptz.get_position()
    print(f"当前位置: pitch={pos[0]:.2f}°, yaw={pos[1]:.2f}°")
    
    del ptz
