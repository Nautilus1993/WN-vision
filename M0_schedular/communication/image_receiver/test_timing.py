#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试耗时统计功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcp_image_receiver import TCPImageReceiver
import time

def test_timing():
    """测试耗时统计功能"""
    print("⏱️  测试耗时统计功能")
    print("=" * 50)
    
    # 创建接收器实例
    receiver = TCPImageReceiver()
    
    try:
        # 连接服务器
        print("1. 连接服务器...")
        if not receiver.connect():
            print("❌ 连接失败")
            return
        
        # 开始接收数据
        print("2. 开始接收数据...")
        if not receiver.start_receiving():
            print("❌ 启动接收失败")
            return
        
        # 发送命令并开始计时
        print("3. 发送命令并开始计时...")
        if not receiver.send_command("set:4k RunWIN:000"):
            print("❌ 发送命令失败")
            return
        
        # 等待接收图像数据
        print("4. 等待接收图像数据...")
        start_wait = time.time()
        timeout = 30  # 30秒超时
        
        while time.time() - start_wait < timeout:
            if receiver.image_count > 0:
                print(f"✅ 接收到第{receiver.image_count}张图像")
                break
            time.sleep(0.1)
        
        if receiver.image_count == 0:
            print("❌ 超时未接收到图像数据")
            return
        
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 断开连接
        receiver.disconnect()
        print("\n🔌 连接已断开")

if __name__ == "__main__":
    test_timing() 