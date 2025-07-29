#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试TCP图像接收器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcp_image_receiver import TCPImageReceiver
import time

def test_image_receiver():
    """测试图像接收器功能"""
    print("🧪 测试TCP图像接收器")
    print("=" * 50)
    
    # 创建接收器实例
    receiver = TCPImageReceiver()
    
    try:
        # 测试连接
        print("1. 测试连接...")
        if receiver.connect():
            print("✅ 连接测试成功")
        else:
            print("❌ 连接测试失败")
            return
        
        # 测试命令发送
        print("\n2. 测试命令发送...")
        if receiver.send_command("set:4k RunWIN:000"):
            print("✅ 命令发送测试成功")
        else:
            print("❌ 命令发送测试失败")
        
        # 测试数据接收（运行5秒）
        print("\n3. 测试数据接收（运行5秒）...")
        if receiver.start_receiving():
            print("✅ 数据接收启动成功")
            
            # 启动命令发送
            receiver.start_command_sending()
            
            # 运行5秒
            start_time = time.time()
            while time.time() - start_time < 5:
                time.sleep(0.1)
                if receiver.image_count > 0:
                    print(f"📷 已接收 {receiver.image_count} 张图像")
                    break
            
            print("⏰ 测试时间结束")
        else:
            print("❌ 数据接收启动失败")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 断开连接
        receiver.disconnect()
        print("\n🔌 连接已断开")
        print("=" * 50)
        print("测试完成")

if __name__ == "__main__":
    test_image_receiver() 