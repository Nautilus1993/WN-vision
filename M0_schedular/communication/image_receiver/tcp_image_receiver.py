#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCP图像接收器
每秒发送set:4k RunWIN:000指令，接收和解析图片数据，保存为时间戳命名的bmp文件
"""

import socket
import time
import threading
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

# 命令相关常量
COMMAND_INTERVAL = 1.0  # 发送命令的间隔时间（秒）
DEFAULT_COMMAND = "set:4k RunWIN:000"


class TCPImageReceiver:
    def __init__(self, server_ip: str = DEFAULT_SERVER_IP, server_port: int = DEFAULT_SERVER_PORT, timeout: int = DEFAULT_TIMEOUT):
        """
        初始化TCP图像接收器
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
        self.is_receiving = False
        self.receive_thread = None
        self.command_thread = None
        self.received_data = bytearray()
        self.image_data = bytearray(DEFAULT_IMAGE_BUFFER_SIZE)
        self.lock = threading.Lock()
        
        # 记录当前图像的实际参数
        self.current_image_width = DEFAULT_IMAGE_WIDTH
        self.current_image_height = DEFAULT_IMAGE_HEIGHT
        
        # 统计信息
        self.packet_count = 0
        self.total_received_bytes = 0
        self.valid_packets = 0
        self.image_count = 0
        
        # 耗时统计
        self.command_send_time = None
        self.image_complete_time = None
        self.save_complete_time = None
        
        # 保存目录
        self.save_directory = "received_images"
        self._ensure_save_directory()
    
    def _ensure_save_directory(self):
        """确保保存目录存在"""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"📁 创建保存目录: {self.save_directory}")
    
    def connect(self) -> bool:
        """连接到TCP服务器"""
        try:
            print(f"🔌 正在连接到 {self.server_ip}:{self.server_port}...")
            
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
            
            # 记录命令发送时间
            self.command_send_time = time.time()
            
            print(f"📤 命令发送成功: {repr(command.strip())}")
            return True
            
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def parse_image_packet(self, data: bytes) -> tuple:
        """解析图像数据包"""
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
            
        return True, fig_width, fig_height, row_id, pack_id, valid_bytes
    
    def process_received_data(self, data: bytes):
        """处理接收到的数据"""
        self.packet_count += 1
        self.total_received_bytes += len(data)
        
        with self.lock:
            self.received_data.extend(data)
            
        data_size = len(self.received_data)
        data_buffer = bytearray(self.received_data)
        save_image = False
        
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
                        save_image = True
                        self.image_count += 1
                        
                        # 记录图像完成时间
                        self.image_complete_time = time.time()
                        
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
        
        # 保存图像
        if save_image:
            self.save_image()
            # 重置计时器，为下一张图片做准备
            self.reset_timers()
    
    def save_image(self):
        """保存图像为BMP文件"""
        try:
            print(f"💾 开始保存第{self.image_count}张图像")
            
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
            
            if non_zero_count == 0:
                print("❌ 错误: 所有数据都是零值，图像将显示为黑色")
                return
            
            # 创建numpy数组
            image_array = np.frombuffer(valid_image_data, dtype=np.uint8)
            
            # 计算像素尺寸
            pixel_width = expected_width_bytes // 3
            
            # 验证数据大小
            expected_array_size = expected_height * pixel_width * 3
            if len(image_array) != expected_array_size:
                print(f"⚠️  数据大小不匹配: {len(image_array)} != {expected_array_size}")
                if len(image_array) > expected_array_size:
                    image_array = image_array[:expected_array_size]
                else:
                    print("❌ 数据不足，无法保存")
                    return
            
            # 重塑为图像
            image = image_array.reshape(expected_height, pixel_width, 3)
            
            # 转换为BGR格式（OpenCV默认格式）
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 生成时间戳文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            filename = f"{timestamp}.bmp"
            filepath = os.path.join(self.save_directory, filename)
            
            # 保存图像
            success = cv2.imwrite(filepath, image_bgr)
            
            if success:
                # 记录保存完成时间
                self.save_complete_time = time.time()
                
                print(f"✅ 图像保存成功:")
                print(f"   文件路径: {filepath}")
                print(f"   图像尺寸: {pixel_width} x {expected_height}")
                print(f"   文件大小: {os.path.getsize(filepath)} 字节")
                
                # 计算总耗时
                if self.command_send_time:
                    total_duration = self.save_complete_time - self.command_send_time
                    save_duration = self.save_complete_time - self.image_complete_time if self.image_complete_time else 0
                    image_duration = self.image_complete_time - self.command_send_time if self.image_complete_time else 0
                    
                    print(f"⏱️  总耗时统计:")
                    print(f"   命令发送到图像完成: {image_duration:.3f}秒")
                    print(f"   图像保存耗时: {save_duration:.3f}秒")
                    print(f"   总耗时: {total_duration:.3f}秒")
                    print(f"⏰ 保存完成时间: {datetime.fromtimestamp(self.save_complete_time).strftime('%H:%M:%S.%f')[:-3]}")
            else:
                print(f"❌ 图像保存失败: {filepath}")
            
        except Exception as e:
            print(f"❌ 保存图像时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_timers(self):
        """重置计时器，为下一张图片做准备"""
        self.command_send_time = None
        self.image_complete_time = None
        self.save_complete_time = None
    
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
    
    def start_command_sending(self):
        """开始定时发送命令"""
        if not self.socket:
            print("❌ 未连接到服务器")
            return False
        
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        
        print(f"📤 开始定时发送命令: {DEFAULT_COMMAND}")
        return True
    
    def _command_loop(self):
        """命令发送循环"""
        try:
            while self.is_receiving and self.socket:
                print("=" * 50)
                self.send_command(DEFAULT_COMMAND)
                time.sleep(COMMAND_INTERVAL)
        except Exception as e:
            print(f"❌ 命令发送循环出错: {e}")
    
    def start(self):
        """启动图像接收器"""
        if self.connect():
            if self.start_receiving():
                self.start_command_sending()
                return True
        return False
    
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    """主函数"""
    print("📡 TCP图像接收器")
    print("=" * 50)
    print("功能: 每秒发送set:4k RunWIN:000指令，接收图片数据并保存为BMP文件")
    print("=" * 50)
    
    try:
        with TCPImageReceiver() as receiver:
            if receiver.start():
                print("\n⏳ 正在运行... (按Ctrl+C退出)")
                print(f"📁 图片将保存到: {receiver.save_directory}")
                
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