#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像调试工具
专门用于读取.bin文件并分析图像数据问题
"""

import numpy as np
import cv2
import os
import glob
from datetime import datetime
import argparse

def read_bin_file(bin_file_path, meta_file_path=None):
    """
    读取.bin文件和对应的.meta文件
    
    Args:
        bin_file_path: .bin文件路径
        meta_file_path: .meta文件路径（可选）
    
    Returns:
        (image_data, metadata) 元组
    """
    print(f"📖 读取文件: {bin_file_path}")
    
    # 读取二进制数据
    try:
        with open(bin_file_path, 'rb') as f:
            image_data = f.read()
        print(f"✅ 成功读取 {len(image_data)} 字节")
    except Exception as e:
        print(f"❌ 读取.bin文件失败: {e}")
        return None, None
    
    # 读取元数据
    metadata = {}
    if meta_file_path and os.path.exists(meta_file_path):
        try:
            with open(meta_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key] = value
            print(f"✅ 成功读取元数据: {len(metadata)} 项")
        except Exception as e:
            print(f"⚠️  读取元数据失败: {e}")
    
    return image_data, metadata

def analyze_image_data(image_data, width_bytes=None, height=None, description=""):
    """
    分析图像数据的详细信息
    
    Args:
        image_data: 图像数据字节数组
        width_bytes: 图像宽度（字节）
        height: 图像高度
        description: 描述信息
    """
    print(f"\n🔍 图像数据分析: {description}")
    print("=" * 60)
    
    # 基本统计
    total_bytes = len(image_data)
    non_zero_bytes = sum(1 for b in image_data if b != 0)
    zero_bytes = total_bytes - non_zero_bytes
    
    print(f"📊 基本统计:")
    print(f"   总字节数: {total_bytes}")
    print(f"   非零字节: {non_zero_bytes}")
    print(f"   零字节数: {zero_bytes}")
    print(f"   非零比例: {non_zero_bytes/total_bytes*100:.2f}%")
    
    # 数值范围分析
    min_val = min(image_data)
    max_val = max(image_data)
    mean_val = sum(image_data) / len(image_data)
    
    print(f"📈 数值范围:")
    print(f"   最小值: {min_val}")
    print(f"   最大值: {max_val}")
    print(f"   平均值: {mean_val:.2f}")
    print(f"   值范围: {max_val - min_val}")
    
    # 按通道分析（假设RGB格式）
    if total_bytes % 3 == 0:
        pixel_count = total_bytes // 3
        print(f"\n🎨 RGB通道分析 (假设RGB格式):")
        print(f"   像素数量: {pixel_count}")
        
        r_values = [image_data[i] for i in range(0, total_bytes, 3)]
        g_values = [image_data[i+1] for i in range(0, total_bytes, 3)]
        b_values = [image_data[i+2] for i in range(0, total_bytes, 3)]
        
        print(f"   R通道: 范围[{min(r_values)}, {max(r_values)}], 平均{sum(r_values)/len(r_values):.2f}")
        print(f"   G通道: 范围[{min(g_values)}, {max(g_values)}], 平均{sum(g_values)/len(g_values):.2f}")
        print(f"   B通道: 范围[{min(b_values)}, {max(b_values)}], 平均{sum(b_values)/len(b_values):.2f}")
        
        # 检查是否有异常值
        r_non_zero = sum(1 for v in r_values if v > 0)
        g_non_zero = sum(1 for v in g_values if v > 0)
        b_non_zero = sum(1 for v in b_values if v > 0)
        
        print(f"   非零像素: R={r_non_zero}, G={g_non_zero}, B={b_non_zero}")
    
    # 前几个字节的详细分析
    print(f"\n🔢 前20字节详细分析:")
    for i in range(min(20, len(image_data))):
        byte_val = image_data[i]
        if i % 3 == 0:
            channel = "R"
        elif i % 3 == 1:
            channel = "G"
        else:
            channel = "B"
        print(f"   字节{i:2d} ({channel}): {byte_val:3d} (0x{byte_val:02X})")
    
    # 检查数据模式
    print(f"\n🔍 数据模式分析:")
    
    # 检查是否全是零
    if non_zero_bytes == 0:
        print("   ❌ 所有数据都是零值！")
        return False
    
    # 检查是否全是相同值
    unique_values = set(image_data)
    if len(unique_values) == 1:
        print(f"   ⚠️  所有数据都是相同值: {list(unique_values)[0]}")
    
    # 检查数据分布
    value_counts = {}
    for val in image_data:
        value_counts[val] = value_counts.get(val, 0) + 1
    
    most_common = max(value_counts.items(), key=lambda x: x[1])
    print(f"   最常见值: {most_common[0]} (出现{most_common[1]}次)")
    
    return True

def test_different_formats(image_data, width_bytes, height, save_dir="debug_images"):
    """
    测试不同的图像格式和转换方式
    
    Args:
        image_data: 原始图像数据
        width_bytes: 图像宽度（字节）
        height: 图像高度
        save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    pixel_width = width_bytes // 3
    
    print(f"\n🧪 测试不同图像格式")
    print("=" * 60)
    
    # 确保数据长度正确
    expected_size = height * pixel_width * 3
    if len(image_data) != expected_size:
        print(f"⚠️  数据大小不匹配: {len(image_data)} != {expected_size}")
        if len(image_data) > expected_size:
            image_data = image_data[:expected_size]
        else:
            print("❌ 数据不足，无法处理")
            return
    
    # 方法1: 直接使用原始数据（假设已经是RGB）
    try:
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image_rgb = image_array.reshape(height, pixel_width, 3)
        
        filename = f"{save_dir}/test1_original_rgb_{timestamp}.png"
        cv2.imwrite(filename, image_rgb)
        print(f"✅ 方法1 - 原始RGB: {filename}")
        print(f"   值范围: [{image_rgb.min()}, {image_rgb.max()}]")
        print(f"   平均值: {image_rgb.mean():.2f}")
        
    except Exception as e:
        print(f"❌ 方法1失败: {e}")
    
    # 方法2: 转换为BGR
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filename = f"{save_dir}/test2_rgb_to_bgr_{timestamp}.png"
        cv2.imwrite(filename, image_bgr)
        print(f"✅ 方法2 - RGB转BGR: {filename}")
        
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
    
    # 方法3: 假设原始数据是BGR，转换为RGB
    try:
        image_bgr_orig = np.frombuffer(image_data, dtype=np.uint8).reshape(height, pixel_width, 3)
        image_rgb_conv = cv2.cvtColor(image_bgr_orig, cv2.COLOR_BGR2RGB)
        filename = f"{save_dir}/test3_bgr_to_rgb_{timestamp}.png"
        cv2.imwrite(filename, image_rgb_conv)
        print(f"✅ 方法3 - BGR转RGB: {filename}")
        
    except Exception as e:
        print(f"❌ 方法3失败: {e}")
    
    # 方法4: 归一化处理
    try:
        image_normalized = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX)
        filename = f"{save_dir}/test4_normalized_{timestamp}.png"
        cv2.imwrite(filename, image_normalized)
        print(f"✅ 方法4 - 归一化: {filename}")
        
    except Exception as e:
        print(f"❌ 方法4失败: {e}")
    
    # 方法5: 直方图均衡化
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        filename = f"{save_dir}/test5_histogram_eq_{timestamp}.png"
        cv2.imwrite(filename, gray_eq)
        print(f"✅ 方法5 - 直方图均衡化: {filename}")
        
    except Exception as e:
        print(f"❌ 方法5失败: {e}")
    
    # 方法6: 调整对比度和亮度
    try:
        alpha = 2.0  # 对比度
        beta = 50    # 亮度
        image_adjusted = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
        filename = f"{save_dir}/test6_adjusted_{timestamp}.png"
        cv2.imwrite(filename, image_adjusted)
        print(f"✅ 方法6 - 对比度亮度调整: {filename}")
        
    except Exception as e:
        print(f"❌ 方法6失败: {e}")
    
    # 方法7: 检查是否数据需要重新排列
    try:
        rearranged_data = bytearray(len(image_data))
        for i in range(0, len(image_data), 3):
            if i + 2 < len(image_data):
                rearranged_data[i] = image_data[i + 2]  # R
                rearranged_data[i + 1] = image_data[i + 1]  # G
                rearranged_data[i + 2] = image_data[i]  # B
        
        image_rearranged = np.frombuffer(rearranged_data, dtype=np.uint8).reshape(height, pixel_width, 3)
        filename = f"{save_dir}/test7_rearranged_{timestamp}.png"
        cv2.imwrite(filename, image_rearranged)
        print(f"✅ 方法7 - 重新排列RGB: {filename}")
        
    except Exception as e:
        print(f"❌ 方法7失败: {e}")
    
    # 方法8: 尝试不同的数据解释方式
    try:
        # 尝试作为灰度图处理
        gray_direct = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width_bytes)
        filename = f"{save_dir}/test8_gray_direct_{timestamp}.png"
        cv2.imwrite(filename, gray_direct)
        print(f"✅ 方法8 - 直接灰度图: {filename}")
        
    except Exception as e:
        print(f"❌ 方法8失败: {e}")

def find_latest_bin_file():
    """查找最新的.bin文件"""
    bin_files = glob.glob("raw_image_*.bin")
    if not bin_files:
        return None, None
    
    # 按修改时间排序
    bin_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_bin = bin_files[0]
    
    # 查找对应的.meta文件
    base_name = latest_bin.replace('.bin', '')
    meta_file = f"{base_name}.meta"
    
    return latest_bin, meta_file if os.path.exists(meta_file) else None

def analyze_bin_file(bin_file_path, meta_file_path=None):
    """分析指定的.bin文件"""
    print(f"🔍 分析文件: {bin_file_path}")
    print("=" * 60)
    
    # 读取文件
    image_data, metadata = read_bin_file(bin_file_path, meta_file_path)
    if image_data is None:
        return False
    
    # 从元数据获取尺寸信息
    width_bytes = None
    height = None
    
    if metadata:
        width_bytes = int(metadata.get('width_bytes', 0))
        height = int(metadata.get('height', 0))
        print(f"📐 从元数据获取尺寸: {width_bytes}字节 × {height}像素")
    
    # 如果没有元数据，尝试推断尺寸
    if not width_bytes or not height:
        print("⚠️  无法从元数据获取尺寸，尝试推断...")
        total_bytes = len(image_data)
        
        # 尝试常见的分辨率
        common_resolutions = [
            (1920, 1080),  # 1920*3=5760字节宽
            (1280, 720),   # 1280*3=3840字节宽
            (640, 480),    # 640*3=1920字节宽
        ]
        
        for w, h in common_resolutions:
            expected_bytes = w * 3 * h
            if total_bytes == expected_bytes:
                width_bytes = w * 3
                height = h
                print(f"✅ 推断尺寸: {width_bytes}字节 × {height}像素")
                break
        
        if not width_bytes or not height:
            print(f"❌ 无法推断尺寸，总字节数: {total_bytes}")
            return False
    
    # 分析数据
    has_valid_data = analyze_image_data(image_data, width_bytes, height, "原始数据")
    
    if has_valid_data:
        # 测试不同格式
        test_different_formats(image_data, width_bytes, height, "debug_images")
        
        print(f"\n✅ 分析完成")
        print(f"📁 调试图像保存在: debug_images/")
    else:
        print(f"\n❌ 数据无效，无法生成图像")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像调试工具 - 分析.bin文件")
    parser.add_argument("-f", "--file", help="指定.bin文件路径")
    parser.add_argument("-m", "--meta", help="指定.meta文件路径")
    parser.add_argument("-l", "--latest", action="store_true", help="分析最新的.bin文件")
    parser.add_argument("--list", action="store_true", help="列出所有.bin文件")
    
    args = parser.parse_args()
    
    print("🔧 图像调试工具")
    print("=" * 60)
    
    # 列出所有.bin文件
    if args.list:
        bin_files = glob.glob("raw_image_*.bin")
        if bin_files:
            print("📁 找到以下.bin文件:")
            for i, f in enumerate(sorted(bin_files, key=lambda x: os.path.getmtime(x), reverse=True), 1):
                size = os.path.getsize(f)
                mtime = datetime.fromtimestamp(os.path.getmtime(f))
                print(f"   {i}. {f} ({size} 字节, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("❌ 未找到.bin文件")
        return
    
    # 确定要分析的文件
    bin_file = None
    meta_file = None
    
    if args.file:
        bin_file = args.file
        meta_file = args.meta
    elif args.latest:
        bin_file, meta_file = find_latest_bin_file()
        if not bin_file:
            print("❌ 未找到.bin文件")
            return
    else:
        # 默认分析最新文件
        bin_file, meta_file = find_latest_bin_file()
        if not bin_file:
            print("❌ 未找到.bin文件")
            print("💡 使用 --list 查看可用文件")
            return
    
    # 分析文件
    success = analyze_bin_file(bin_file, meta_file)
    
    if success:
        print("\n✅ 分析完成")
    else:
        print("\n❌ 分析失败")

if __name__ == "__main__":
    main() 