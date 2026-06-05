#!/usr/bin/env python3
"""蜂鸣器控制类 - 使用GPIO控制有源蜂鸣器.

有源蜂鸣器：高电平响，低电平不响。
"""

import time
import threading
import Hobot.GPIO as GPIO


class Buzzer:
    """有源蜂鸣器控制类.
    
    Attributes:
        pin: GPIO引脚编号（BOARD编码）
    
    使用示例:
        ```python
        buzzer = Buzzer(pin=33)
        buzzer.beep(count=2, duration=0.15, interval=0.15)  # 响2声
        buzzer.cleanup()
        ```
    """

    def __init__(self, pin: int = 33):
        """初始化蜂鸣器GPIO.
        
        Args:
            pin: GPIO引脚编号（BOARD编码），默认33
        """
        self.pin = pin
        self._gpio_setup = False
        self._setup_gpio()

    def _setup_gpio(self):
        """配置GPIO引脚为输出模式，初始低电平（不响）."""
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._gpio_setup = True

    def on(self):
        """蜂鸣器响 - 设置高电平."""
        if not self._gpio_setup:
            self._setup_gpio()
        GPIO.output(self.pin, GPIO.HIGH)#

    def off(self):
        """蜂鸣器停 - 设置低电平."""
        if self._gpio_setup:
            GPIO.output(self.pin, GPIO.LOW)

    def beep(self, count: int = 1, duration: float = 0.15, interval: float = 0.15):
        """蜂鸣器响指定次数（阻塞调用）.
        
        Args:
            count: 响的次数
            duration: 每次响的持续时间（秒）
            interval: 两次响之间的间隔时间（秒）
        """
        for i in range(count):
            self.on()
            time.sleep(duration)
            self.off()
            if i < count - 1:
                time.sleep(interval)

    def beep_async(self, count: int = 1, duration: float = 0.15, interval: float = 0.15):
        """蜂鸣器响指定次数（非阻塞，在后台线程执行）.
        
        Args:
            count: 响的次数
            duration: 每次响的持续时间（秒）
            interval: 两次响之间的间隔时间（秒）
        """
        t = threading.Thread(
            target=self.beep,
            args=(count, duration, interval),
            daemon=True
        )
        t.start()

    def cleanup(self):
        """清理GPIO资源，确保蜂鸣器停止."""
        self.off()
        if self._gpio_setup:
            GPIO.cleanup(self.pin)
            self._gpio_setup = False

    def __del__(self):
        """析构函数，确保清理."""
        try:
            self.cleanup()
        except Exception:
            pass


if __name__ == '__main__':
    buzzer = Buzzer(pin=40)
    print("蜂鸣器响1声")
    buzzer.beep(count=1)
    time.sleep(0.5)
    
    print("蜂鸣器响2声")
    buzzer.beep(count=2)
    time.sleep(0.5)
    
    print("蜂鸣器响3声")
    buzzer.beep(count=3)
    
    buzzer.cleanup()
    print("完成")
