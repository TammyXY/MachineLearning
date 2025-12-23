import pandas as pd
import numpy as np

# 加载预测结果文件
pred_file_path = 'result_Smart_Model_Selection.csv'
pred_data = pd.read_csv(pred_file_path)

# 加载真实值数据集
true_file_path = "C:/ProgramData/anaconda3/envs/pythonProject2/machinelearning/002-数据集/数据集（含真实值）/modified_数据集Time_Series662.dat"
true_data = pd.read_csv(true_file_path)

# 定义目标列名
target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
true_values = true_data[target_columns]

# 将预测值字符串转换为数值列表
pred_data['Predicted_Value'] = pred_data['Predicted_Value'].apply(lambda x: list(map(float, x.split())))

# 将真实值转换为列表
true_values_list = true_values.values.tolist()

# 确保预测值和真实值的行数相同
assert len(pred_data) == len(true_values_list), "预测值和真实值的行数不匹配，请检查数据"

# 计算每行预测值与真实值的差值
errors = []
for pred, true in zip(pred_data['Predicted_Value'], true_values_list):
    error = np.abs(np.array(pred) - np.array(true))
    errors.append(error)

# 转换为numpy数组便于计算
errors = np.array(errors)

# 计算每个特征的平均误差
mean_errors = np.mean(errors, axis=0)

# 计算总体平均误差
overall_mean_error = np.mean(errors)

# 输出结果
feature_names = target_columns
print("每个特征的平均误差：")
for feature, error in zip(feature_names, mean_errors):
    print(f"{feature}: {error:.4f}")

print(f"\n总体平均误差: {overall_mean_error:.4f}")
