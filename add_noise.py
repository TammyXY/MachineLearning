# 时间：2024年4月25日  Date： April 25, 2024
# 文件名称 Filename： 01-add_noise.py
# 编码实现 Coding by： Yunfei Gui, Jinhuan Luo 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师

# coding=utf-8
# 用于添加噪声的代码

import pandas as pd
import numpy as np
'''
0：未添加噪声，表示正确数据
1：添加噪声，表示错误数据
'''
def add_noise_to_csv_448(output_file, noise_probability, noise_range, columns):
    # 读取输入文件为 DataFrame
    df = pd.read_csv('数据集Time_Series_448.dat', header=0)

    # 遍历指定的列
    for col in columns:
        # 根据噪声概率随机决定是否添加噪声
        mask = np.random.choice([True, False], size=len(df), p=[noise_probability, 1 - noise_probability])
        noise = np.random.uniform(noise_range[0], noise_range[1], size=len(df))  # 生成噪声

        # 添加新的错误标志列
        error_column = 'Error_' + col
        df[error_column] = df[col].copy() # 复制原始列的值
        df.loc[mask, error_column] += noise[mask] # 添加噪声

    # 将处理后的 DataFrame 写入输出文件 output_file
    df.to_csv(output_file, index=False)


# 调用函数，传入输入文件路径、输出文件路径、噪声概率、噪声范围和要添加噪声的列
output_file_448 = 'modified_数据集Time_Series448_detail.dat'
noise_probability_448 = 0.5  # 噪声概率为 0.5
noise_range_448 = (-2.0, 5.0)  # 噪声范围为 -2.0 到 3.0
columns_448 =  ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth', 'RECORD']   # 要添加噪声的列

# 调用添加噪声的函数
# add_noise_to_csv_448(output_file_448, noise_probability_448, noise_range_448, columns_448)


def add_noise_to_csv_660(output_file, noise_probability, noise_range, columns):
    # 读取输入文件为 DataFrame
    df = pd.read_csv('数据集Time_Series_660.dat', header=0)
    # 创建一个新的 DataFrame 用于记录每行数据是否发生变化

    # 遍历指定的列
    for col in columns:
        # 根据噪声概率随机决定是否添加噪声
        mask = np.random.choice([True, False], size=len(df), p=[noise_probability, 1 - noise_probability])
        noise = np.random.uniform(noise_range[0], noise_range[1], size=len(df))  # 生成噪声

        # 添加新的错误标志列
        error_column = 'Error_' + col
        df[error_column] = df[col].copy() # 复制原始列的值
        df.loc[mask, error_column] += noise[mask] # 添加噪声

    # 将处理后的 DataFrame 写入输出文件 output_file
    df.to_csv(output_file, index=False)


# 调用函数，传入输入文件路径、输出文件路径、噪声概率、噪声范围和要添加噪声的列
output_file_660 = 'modified_数据集Time_Series660_detail.dat'
noise_probability_660 = 0.4  # 噪声概率为 0.4
noise_range_660 = (-2.0, 4.0)  # 噪声范围为 -2.0 到 4.0
columns_660 =  ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth', 'RECORD']  # 要添加噪声的列

# 调用添加噪声的函数
# add_noise_to_csv_660(output_file_660, noise_probability_660, noise_range_660, columns_660)

## 测试集
def add_noise_to_csv(output_file, noise_probability, noise_range, columns):
    # 读取输入文件为 DataFrame
    df = pd.read_csv('数据集Time_Series_672.dat', header=0)
    # 创建一个新的 DataFrame 用于记录每行数据是否发生变化

    # 遍历指定的列
    for col in columns:
        # 根据噪声概率随机决定是否添加噪声
        mask = np.random.choice([True, False], size=len(df), p=[noise_probability, 1 - noise_probability])
        noise = np.random.uniform(noise_range[0], noise_range[1], size=len(df)).astype(float)  # 生成噪声

        # 添加新的错误标志列
        error_column = 'Error_' + col
        df[error_column] = df[col].copy() # 复制原始列的值
        df.loc[mask, error_column] = df.loc[mask, error_column].astype(float) + noise[mask] # 添加噪声

    # 将处理后的 DataFrame 写入输出文件 output_file
    df.to_csv(output_file, index=False)


# 调用函数，传入输入文件路径、输出文件路径、噪声概率、噪声范围和要添加噪声的列
output_file_662 = 'modified_数据集Time_Series672.dat'
noise_probability_662 = 0.5  # 噪声概率为 0.4
noise_range_662 = (-2.0, 2.0)  # 噪声范围为 -2.0 到 2.0
columns_662 =  ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']  # 要添加噪声的列

# 调用添加噪声的函数
add_noise_to_csv(output_file_662, noise_probability_662, noise_range_662, columns_662)