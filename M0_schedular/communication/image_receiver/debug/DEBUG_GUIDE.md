# 图像调试指南

## 问题描述

当使用 `tcp_command_sender_debug.py` 接收图像时，Normalized图像能显示一些内容，但BGR和原始图像都是黑的。

## 解决方案

我们重新设计了调试流程，将功能分离：

1. **`tcp_command_sender_debug.py`** - 专注于网络数据接收，将原始数据保存为 `.bin` 文件
2. **`image_debug_tool.py`** - 专注于读取 `.bin` 文件并分析数据问题

## 使用步骤

### 步骤1: 接收图像数据

运行调试版本的TCP接收器，它会自动保存原始数据：

```bash
python3 tcp_command_sender_debug.py
```

程序会：
- 连接到服务器并接收图像数据
- 将原始图像数据保存为 `raw_image_X_时间戳.bin` 文件
- 将元数据保存为 `raw_image_X_时间戳.meta` 文件

### 步骤2: 分析图像数据

使用图像调试工具分析保存的数据：

```bash
# 分析最新的.bin文件
python3 image_debug_tool.py

# 或者指定特定文件
python3 image_debug_tool.py -f raw_image_1_20241201_143022_123.bin

# 列出所有可用的.bin文件
python3 image_debug_tool.py --list
```

## 调试工具功能

`image_debug_tool.py` 会：

1. **读取并分析原始数据**
   - 统计非零字节数
   - 分析数值范围
   - 检查RGB通道分布

2. **测试8种不同的显示方法**
   - 原始RGB格式
   - RGB转BGR格式
   - BGR转RGB格式
   - 归一化处理
   - 直方图均衡化
   - 对比度亮度调整
   - RGB重新排列
   - 直接灰度图

3. **生成诊断报告**
   - 识别常见问题
   - 提供解决建议

## 常见问题诊断

### 问题1: 数据全为零值
**症状**: 所有字节都是0
**可能原因**: 
- 服务器未正确发送图像数据
- 命令格式错误
- 网络连接问题

### 问题2: 数据全为相同值
**症状**: 所有字节都是相同值（如128）
**可能原因**:
- 图像数据编码格式错误
- 数据包解析问题

### 问题3: 数据范围过小
**症状**: 数值范围很小（如0-10）
**可能原因**:
- 图像源问题
- 需要图像增强处理

### 问题4: 数据大小不匹配
**症状**: 数据长度与预期不符
**可能原因**:
- 图像尺寸参数错误
- 数据包不完整

## 文件说明

### 生成的文件

- `raw_image_X_时间戳.bin` - 原始图像数据
- `raw_image_X_时间戳.meta` - 元数据信息
- `debug_images/` - 调试图像目录
  - `test1_original_rgb_*.png` - 原始RGB格式
  - `test2_rgb_to_bgr_*.png` - RGB转BGR
  - `test3_bgr_to_rgb_*.png` - BGR转RGB
  - `test4_normalized_*.png` - 归一化处理
  - `test5_histogram_eq_*.png` - 直方图均衡化
  - `test6_adjusted_*.png` - 对比度亮度调整
  - `test7_rearranged_*.png` - RGB重新排列
  - `test8_gray_direct_*.png` - 直接灰度图

### 元数据格式

```
width_bytes=5760
height=1080
total_bytes=6220800
non_zero_bytes=1234567
timestamp=20241201_143022_123
image_count=1
```

## 使用示例

```bash
# 1. 接收图像数据
python3 tcp_command_sender_debug.py

# 2. 查看生成的.bin文件
python3 image_debug_tool.py --list

# 3. 分析最新文件
python3 image_debug_tool.py

# 4. 分析特定文件
python3 image_debug_tool.py -f raw_image_1_20241201_143022_123.bin
```

## 预期结果

如果调试工具能生成可见的图像，说明：
1. 数据接收正常
2. 问题在于显示格式或处理方式

如果所有方法都生成黑色图像，说明：
1. 原始数据有问题
2. 需要检查网络连接和服务器配置

## 下一步

根据调试结果：
1. 如果某个方法能显示图像，使用该方法修改显示代码
2. 如果所有方法都失败，检查网络和服务器配置
3. 如果数据全为零，检查命令格式和服务器响应 