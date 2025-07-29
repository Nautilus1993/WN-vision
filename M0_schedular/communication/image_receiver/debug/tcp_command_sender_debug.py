#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCP命令发送器 - 调试版本
专门用于诊断图像显示问题
"""

import socket
import time
import argparse
import sys
import threading
import struct
from typing import Optional, Callable
import numpy as np
import cv2
from datetime import datetime
import os

# 全局常量定义
# 图像协议相关常量
IMAGE_HEADER_LENGTH = 10  # 图像数据包头长度
IMAGE_CHUNK_LENGTH = 1436  # 图像数据块长度
IMAGE_CHUNK_DATA_LENGTH = IMAGE_CHUNK_LENGTH - 2  # 图像数据块中实际数据长度 (1436 - 2)
IMAGE_PACKET_TAIL_LENGTH = 2  # 图像数据包尾部长度

# 图像包头标识
IMAGE_HEADER_START_BYTE = 0x1A  # 图像包头起始字节
IMAGE_HEADER_END_BYTE = 0xCF    # 图像包头结束字节

# 默认图像尺寸
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
DEFAULT_IMAGE_BYTES_PER_PIXEL = 3  # RGB格式，每像素3字节
DEFAULT_IMAGE_BUFFER_SIZE = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * DEFAULT_IMAGE_BYTES_PER_PIXEL * 2

# 网络相关常量
DEFAULT_SERVER_IP = "10.2.3.250"
DEFAULT_SERVER_PORT = 8080
DEFAULT_TIMEOUT = 5
DEFAULT_RECEIVE_BUFFER_SIZE = 4096

# 调试相关常量
DEBUG_SAMPLE_PIXELS_COUNT = 10
DEBUG_SAMPLE_BYTES_COUNT = 10
DEBUG_PIXEL_CHECK_COUNT = 100
DEBUG_STATISTICS_INTERVAL = 100
DEBUG_DETAILED_PACKETS_COUNT = 5
IS_SAVE_IMAGE = True


class TCPCommandSenderDebug:
    def __init__(self, server_ip: str = DEFAULT_SERVER_IP, server_port: int = DEFAULT_SERVER_PORT, timeout: int = DEFAULT_TIMEOUT):
        """
        初始化TCP命令发送器 - 调试版本
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
        self.is_receiving = False
        self.receive_thread = None
        self.received_data = bytearray()
        self.image_data = bytearray(DEFAULT_IMAGE_BUFFER_SIZE)
        self.data_handler = None
        self.lock = threading.Lock()
        
        # 记录当前图像的实际参数
        self.current_image_width = DEFAULT_IMAGE_WIDTH
        self.current_image_height = DEFAULT_IMAGE_HEIGHT
        
        # 调试统计
        self.packet_count = 0
        self.total_received_bytes = 0
        self.valid_packets = 0
        self.image_count = 0
    
    def connect(self) -> bool:
        """连接到TCP服务器"""
        try:
            print(f"正在连接到 {self.server_ip}:{self.server_port}...")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_ip, self.server_port))
            
            print("✅ 连接成功!")
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def send_command(self, command: str) -> bool:
        """发送命令到服务器"""
        if not self.socket:
            print("❌ 未连接到服务器")
            return False
        
        try:
            if not command.endswith('\n'):
                command += '\n'
            
            command_bytes = command.encode('utf-8')
            self.socket.send(command_bytes)
            
            print(f"✅ 命令发送成功: {repr(command.strip())}")
            return True
            
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def parse_image_packet(self, data: bytes) -> tuple:
        """解析图像数据包 - 调试版本"""
        if len(data) < IMAGE_HEADER_LENGTH:
            return False, 0, 0, 0, 0, 0
            
        # 检查包头 0x1A 0xCF
        if data[0] != IMAGE_HEADER_START_BYTE or data[1] != IMAGE_HEADER_END_BYTE:
            return False, 0, 0, 0, 0, 0
            
        # 解析图像参数
        fig_width = (data[2] << 8) + data[3]
        fig_height = (data[4] << 8) + data[5]
        row_id = (data[6] << 8) + data[7]
        pack_id = (data[8] << 8) + data[9]
        
        # 计算有效字节数
        fig_offset = (fig_width * row_id) + (IMAGE_CHUNK_DATA_LENGTH * pack_id)
        valid_bytes = fig_width - (IMAGE_CHUNK_DATA_LENGTH * pack_id)
        
        if valid_bytes > IMAGE_CHUNK_DATA_LENGTH:
            valid_bytes = IMAGE_CHUNK_DATA_LENGTH
        
        # 调试信息
        if self.valid_packets < 5:  # 只打印前5个包的详细信息
            print(f"🔍 数据包解析:")
            print(f"   包头: 0x{data[0]:02X} 0x{data[1]:02X}")
            print(f"   图像尺寸: {fig_width} x {fig_height}")
            print(f"   位置: row_id={row_id}, pack_id={pack_id}")
            print(f"   偏移: {fig_offset}, 有效字节: {valid_bytes}")
            
        return True, fig_width, fig_height, row_id, pack_id, valid_bytes
    
    def convert_format(self, source: bytes) -> bytes:
        """转换图像数据格式 - 调试版本"""
        length = len(source)
        if length % 3 != 0:
            print(f"⚠️  警告: 数据长度({length})不是3的倍数")
            return source
            
        group_count = length // 3
        result = bytearray(length)
        
        for i in range(group_count):
            # 转换RGB顺序：BGR -> RGB
            result[i * 3] = source[i * 3 + 2]     # R
            result[i * 3 + 1] = source[i * 3 + 1] # G  
            result[i * 3 + 2] = source[i * 3]     # B
            
        # 调试: 检查前几个像素的值
        if group_count > 0:
            sample_pixels = min(DEBUG_SAMPLE_PIXELS_COUNT, group_count)
            pixel_values = []
            for i in range(sample_pixels):
                r = result[i * 3]
                g = result[i * 3 + 1] 
                b = result[i * 3 + 2]
                pixel_values.append(f"({r},{g},{b})")
            
            print(f"🎨 前{sample_pixels}个像素RGB值: {', '.join(pixel_values)}")
            
            # 统计非零像素
            non_zero_count = sum(1 for i in range(0, min(DEBUG_PIXEL_CHECK_COUNT * 3, len(result)), 3) 
                               if result[i] > 0 or result[i+1] > 0 or result[i+2] > 0)
            print(f"📊 前{DEBUG_PIXEL_CHECK_COUNT}个像素中非零像素数量: {non_zero_count}/{DEBUG_PIXEL_CHECK_COUNT}")
            
        return bytes(result)
    
    def process_received_data(self, data: bytes):
        """处理接收到的数据 - 调试版本"""
        self.packet_count += 1
        self.total_received_bytes += len(data)
        
        if self.packet_count % DEBUG_STATISTICS_INTERVAL == 0:
            print(f"📊 统计: 已接收 {self.packet_count} 个数据包, {self.total_received_bytes} 字节")
        
        with self.lock:
            self.received_data.extend(data)
            
        data_size = len(self.received_data)
        data_buffer = bytearray(self.received_data)
        show_image = False
        
        while data_size > IMAGE_HEADER_LENGTH:
            # 解析数据包
            is_valid, fig_width, fig_height, row_id, pack_id, valid_bytes = self.parse_image_packet(data_buffer)
            
            if is_valid:
                self.valid_packets += 1
                
                # 更新当前图像参数
                self.current_image_width = fig_width
                self.current_image_height = fig_height
                
                # 计算图像偏移
                fig_offset = (fig_width * row_id) + (IMAGE_CHUNK_DATA_LENGTH * pack_id)
                
                if data_size >= (IMAGE_HEADER_LENGTH + valid_bytes + IMAGE_PACKET_TAIL_LENGTH):
                    # 提取图像数据段
                    segment = data_buffer[IMAGE_HEADER_LENGTH:IMAGE_HEADER_LENGTH + valid_bytes]
                    
                    # 检查数据内容
                    if self.valid_packets <= DEBUG_DETAILED_PACKETS_COUNT:
                        non_zero_bytes = sum(1 for b in segment if b != 0)
                        print(f"📦 数据段: {len(segment)} 字节, 非零字节: {non_zero_bytes}")
                        
                        if len(segment) >= DEBUG_SAMPLE_BYTES_COUNT:
                            sample_bytes = [f"0x{b:02X}" for b in segment[:DEBUG_SAMPLE_BYTES_COUNT]]
                            print(f"🔢 前{DEBUG_SAMPLE_BYTES_COUNT}字节: {' '.join(sample_bytes)}")
                    
                    # 将数据写入图像缓冲区
                    if fig_offset + valid_bytes <= len(self.image_data):
                        self.image_data[fig_offset:fig_offset + valid_bytes] = segment
                    else:
                        print(f"⚠️  数据越界: offset={fig_offset}, bytes={valid_bytes}, buffer_size={len(self.image_data)}")
                    
                    # 移除已处理的数据
                    data_buffer = data_buffer[IMAGE_HEADER_LENGTH + valid_bytes + IMAGE_PACKET_TAIL_LENGTH:]
                    
                    # 检查是否完成一帧图像
                    expected_size = fig_width * fig_height
                    if (fig_offset + valid_bytes) >= expected_size:
                        show_image = True
                        self.image_count += 1
                        print(f"📷 完成第{self.image_count}张图像: {fig_width}x{fig_height}")
                else:
                    break
            else:
                # 检查是否是文本响应
                try:
                    text_data = data_buffer.decode('utf-8', errors='ignore')
                    if '\n' in text_data:
                        lines = text_data.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                print(f"📨 收到文本响应: {repr(line.strip())}")
                        
                        last_newline = text_data.rfind('\n')
                        if last_newline != -1:
                            bytes_to_remove = last_newline + 1
                            data_buffer = data_buffer[bytes_to_remove:]
                    else:
                        data_buffer = data_buffer[1:]
                except UnicodeDecodeError:
                    data_buffer = data_buffer[1:]
                
            data_size = len(data_buffer)
        
        # 更新接收缓冲区
        with self.lock:
            self.received_data = bytearray(data_buffer)
        
        # 保存原始数据
        if show_image:
            self.save_raw_data()
    
    def save_raw_data(self):
        """保存原始图像数据到.bin文件"""
        try:
            print(f"\n💾 保存第{self.image_count}张图像的原始数据")
            
            # 计算实际需要的图像数据大小
            expected_width_bytes = self.current_image_width
            expected_height = self.current_image_height
            expected_total_bytes = expected_width_bytes * expected_height
            
            print(f"📐 图像参数: {expected_width_bytes} × {expected_height} = {expected_total_bytes} 字节")
            
            # 截取有效的图像数据部分
            valid_image_data = self.image_data[:expected_total_bytes]
            
            # 检查缓冲区数据
            non_zero_count = sum(1 for b in valid_image_data if b != 0)
            print(f"📊 有效数据统计: {len(valid_image_data)} 字节, 非零字节: {non_zero_count}")
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"raw_image_{self.image_count}_{timestamp}.bin"
            
            # 保存原始数据
            with open(filename, 'wb') as f:
                f.write(valid_image_data)
            
            print(f"✅ 原始数据已保存: {filename}")
            print(f"📁 文件大小: {os.path.getsize(filename)} 字节")
            
            # 保存元数据信息
            meta_filename = f"raw_image_{self.image_count}_{timestamp}.meta"
            with open(meta_filename, 'w') as f:
                f.write(f"width_bytes={expected_width_bytes}\n")
                f.write(f"height={expected_height}\n")
                f.write(f"total_bytes={expected_total_bytes}\n")
                f.write(f"non_zero_bytes={non_zero_count}\n")
                f.write(f"timestamp={timestamp}\n")
                f.write(f"image_count={self.image_count}\n")
            
            print(f"✅ 元数据已保存: {meta_filename}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 保存原始数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def start_receiving(self):
        """开始接收数据"""
        if not self.socket:
            print("❌ 未连接到服务器")
            return False
        
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        
        print("📡 开始接收数据...")
        return True
    
    def _receive_loop(self):
        """数据接收循环"""
        try:
            while self.is_receiving and self.socket:
                try:
                    self.socket.settimeout(0.1)
                    data = self.socket.recv(DEFAULT_RECEIVE_BUFFER_SIZE)
                    
                    if not data:
                        print("🔌 服务器关闭了连接")
                        break
                    
                    self.process_received_data(data)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"❌ 接收数据时出错: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ 数据接收循环出错: {e}")
        finally:
            self.is_receiving = False
            print("📡 数据接收已停止")
    
    def send_and_keep_alive(self, command: str):
        """发送命令并保持连接活跃"""
        if self.send_command(command):
            if not self.is_receiving:
                self.start_receiving()
    
    def disconnect(self):
        """断开连接"""
        self.is_receiving = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1)
        
        if self.socket:
            try:
                self.socket.close()
                print("🔌 连接已断开")
            except Exception as e:
                print(f"⚠️ 断开连接时出错: {e}")
            finally:
                self.socket = None
        
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    """主函数 - 调试模式"""
    parser = argparse.ArgumentParser(description="TCP命令发送器 - 调试版本")
    parser.add_argument("-i", "--ip", default=DEFAULT_SERVER_IP, help="服务器IP地址")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_SERVER_PORT, help="服务器端口")
    parser.add_argument("-c", "--command", default="set:4k RunWIN:000", help="要发送的命令")
    
    args = parser.parse_args()
    
    print("🔍 TCP命令发送器 - 调试版本")
    print("=" * 50)
    print("此版本会显示详细的调试信息，帮助诊断图像显示问题")
    print("=" * 50)
    
    try:
        with TCPCommandSenderDebug(args.ip, args.port) as sender:
            if sender.connect():
                print(f"\n📡 发送命令: {args.command}")
                sender.send_and_keep_alive(args.command)
                
                print("\n⏳ 等待图像数据... (按Ctrl+C退出)")
                while True:
                    time.sleep(1)
                    
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 