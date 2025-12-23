import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import time
from tqdm import tqdm
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


class SmartModelEnsemble:
    def __init__(self):
        self.models = {}
        self.best_model_for_target = {}
        self.feature_importance = {}

    def create_target_specific_features(self, data, feature_cols):
        """为不同目标变量创建专门的特征工程"""
        df = data.copy()

        # 基础特征
        base_features = feature_cols.copy()

        # === 针对前4个特征（RF表现好）的特征工程 ===
        # 这些特征通常有较强的时序依赖和线性关系

        # 滞后特征
        for col in feature_cols:
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                base_features.append(f'{col}_lag_{lag}')

        # 滚动统计 - RF擅长处理这类特征
        for col in feature_cols:
            for window in [3, 5, 10]:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                base_features.extend([
                    f'{col}_roll_mean_{window}',
                    f'{col}_roll_std_{window}'
                ])

        # === 针对后2个特征（LGBM表现好）的特殊特征 ===
        # 这些特征（信号强度）通常有复杂的非线性关系

        # 信号强度专门特征
        signal_features = []

        # 信号比率和交互
        df['H2O_CO2_sig_ratio'] = df['Error_H2O_sig_strgth'] / (df['Error_CO2_sig_strgth'] + 1e-8)
        df['sig_strength_sum'] = df['Error_H2O_sig_strgth'] + df['Error_CO2_sig_strgth']
        signal_features.extend(['H2O_CO2_sig_ratio', 'sig_strength_sum'])

        # 信号波动特征 - LGBM擅长学习这类复杂模式
        for col in ['Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']:
            for window in [5, 10, 20]:
                # 波动率
                df[f'{col}_volatility_{window}'] = df[col].rolling(window).std() / (
                            df[col].rolling(window).mean() + 1e-8)
                # 动量特征
                df[f'{col}_momentum_{window}'] = df[col] - df[col].shift(window)
                signal_features.extend([
                    f'{col}_volatility_{window}',
                    f'{col}_momentum_{window}'
                ])

        # 信号变化率
        df['H2O_sig_change_rate'] = df['Error_H2O_sig_strgth'].pct_change()
        df['CO2_sig_change_rate'] = df['Error_CO2_sig_strgth'].pct_change()
        signal_features.extend(['H2O_sig_change_rate', 'CO2_sig_change_rate'])

        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        # 返回所有特征
        all_features = base_features + signal_features

        return df, all_features, base_features, signal_features

    def select_best_model_per_target(self, X_train, y_train, X_val, y_val, target_columns):
        """为每个目标变量选择最佳模型"""
        print("为每个目标变量选择最佳模型...")

        # 模型参数
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 217,
            'verbose': 0
        }

        lgb_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 64,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 217,
            'verbose': -1
        }

        best_models = {}
        validation_results = {}

        for i, target in enumerate(tqdm(target_columns, desc="模型选择")):
            # 分别用RF和LGBM训练并验证
            rf_model = RandomForestRegressor(**rf_params)
            lgb_model = LGBMRegressor(**lgb_params)

            rf_model.fit(X_train, y_train[:, i])
            lgb_model.fit(X_train, y_train[:, i])

            rf_pred = rf_model.predict(X_val)
            lgb_pred = lgb_model.predict(X_val)

            # 计算验证集性能
            rf_r2 = r2_score(y_val[:, i], rf_pred)
            lgb_r2 = r2_score(y_val[:, i], lgb_pred)

            rf_mae = mean_absolute_error(y_val[:, i], rf_pred)
            lgb_mae = mean_absolute_error(y_val[:, i], lgb_pred)

            # 选择最佳模型
            if rf_r2 > lgb_r2 and rf_mae < lgb_mae:
                best_model = rf_model
                best_model_name = 'RF'
                best_score = rf_r2
            else:
                best_model = lgb_model
                best_model_name = 'LGBM'
                best_score = lgb_r2

            best_models[target] = best_model
            self.best_model_for_target[target] = best_model_name

            validation_results[target] = {
                'RF_R2': rf_r2,
                'LGBM_R2': lgb_r2,
                'RF_MAE': rf_mae,
                'LGBM_MAE': lgb_mae,
                'Best_Model': best_model_name,
                'Best_Score': best_score
            }

            print(f"  {target}: {best_model_name} (RF_R2: {rf_r2:.4f}, LGBM_R2: {lgb_r2:.4f})")

        self.models = best_models
        return validation_results

    def train_final_models(self, X_train, y_train, target_columns, feature_sets=None):
        """使用选定的最佳模型进行最终训练"""
        print("训练最终模型...")

        # 模型参数（使用完整数据训练）
        rf_final_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 217,
            'verbose': 0
        }

        lgb_final_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 10,
            'num_leaves': 128,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'random_state': 217,
            'verbose': -1
        }

        # 为信号强度特征使用专门的特征集
        if feature_sets:
            signal_targets = ['H2O_sig_strgth', 'CO2_sig_strgth']
            signal_features = feature_sets['signal_features']
            base_features = feature_sets['base_features']
        else:
            signal_targets = []
            signal_features = base_features = X_train.columns.tolist()

        for i, target in enumerate(tqdm(target_columns, desc="最终训练")):
            # 选择特征集
            if target in signal_targets and feature_sets:
                X_train_selected = X_train[signal_features]
            else:
                X_train_selected = X_train[base_features]

            # 根据之前的选择训练最终模型
            if self.best_model_for_target[target] == 'RF':
                model = RandomForestRegressor(**rf_final_params)
            else:
                model = LGBMRegressor(**lgb_final_params)

            model.fit(X_train_selected, y_train[:, i])
            self.models[target] = model

        return self.models

    def predict_with_best_models(self, X_test, target_columns, feature_sets=None):
        """使用最佳模型进行预测"""
        print("使用最佳模型进行预测...")

        predictions = np.zeros((len(X_test), len(target_columns)))

        for i, target in enumerate(target_columns):
            # 选择特征集
            if target in ['H2O_sig_strgth', 'CO2_sig_strgth'] and feature_sets:
                X_test_selected = X_test[feature_sets['signal_features']]
            else:
                X_test_selected = X_test[feature_sets['base_features']]

            # 使用对应的最佳模型预测
            pred = self.models[target].predict(X_test_selected)
            predictions[:, i] = pred

            print(f"  {target}: {self.best_model_for_target[target]}")

        return predictions

    def evaluate_performance(self, y_true, predictions, target_columns):
        """评估性能"""
        print("\n" + "=" * 70)
        print("智能模型选择性能评估")
        print("=" * 70)

        results = {}
        overall_errors = []

        print("\n各特征详细结果:")
        for i, target in enumerate(target_columns):
            r2 = r2_score(y_true[:, i], predictions[:, i])
            mae = mean_absolute_error(y_true[:, i], predictions[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], predictions[:, i]))

            results[target] = {
                'R2': r2,
                'MAE': mae,
                'RMSE': rmse,
                'Model': self.best_model_for_target[target]
            }

            overall_errors.append(mae)

            print(f"  {target:25} | {self.best_model_for_target[target]:4} | "
                  f"R²: {r2:7.4f} | MAE: {mae:9.6f} | RMSE: {rmse:9.6f}")

        # 总体指标
        overall_mae = np.mean(overall_errors)
        overall_r2 = r2_score(y_true, predictions, multioutput='variance_weighted')

        results['overall'] = {
            'MAE': overall_mae,
            'R2': overall_r2
        }

        print(f"\n总体性能: MAE = {overall_mae:.6f}, R² = {overall_r2:.4f}")

        # 模型使用统计
        rf_count = sum(1 for model in self.best_model_for_target.values() if model == 'RF')
        lgb_count = sum(1 for model in self.best_model_for_target.values() if model == 'LGBM')

        print(f"\n模型使用统计:")
        print(f"  Random Forest: {rf_count} 个特征")
        print(f"  LightGBM:      {lgb_count} 个特征")

        return results, overall_mae


def main():
    start_time = time.time()

    print("=== 智能模型选择集成方案 ===")
    print("策略: 前4个特征用RF，后2个特征用LGBM")

    # 加载数据
    print("1. 加载数据...")
    train_data = pd.read_csv(r'C:/ProgramData/anaconda3/envs/pythonProject2/machinelearning/002-数据集/加噪数据集/modified_数据集Time_Series661_detail.dat')
    test_data = pd.read_csv(r'C:/ProgramData/anaconda3/envs/pythonProject2/machinelearning/002-数据集/加噪数据集/modified_数据集Time_Series662_detail.dat')

    # 数据采样（可选）
    # train_data = train_data.iloc[::2].reset_index(drop=True)
    # test_data = test_data.iloc[::2].reset_index(drop=True)

    # 定义特征和目标
    feature_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                       'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
                      'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

    # 前4个目标（RF表现好）
    rf_targets = target_columns[:4]
    # 后2个目标（LGBM表现好）
    lgb_targets = target_columns[4:]

    print(f"RF目标: {rf_targets}")
    print(f"LGBM目标: {lgb_targets}")

    # 创建智能集成模型
    smart_ensemble = SmartModelEnsemble()

    # 特征工程
    print("2. 目标特定的特征工程...")
    train_enhanced, all_features, base_features, signal_features = smart_ensemble.create_target_specific_features(
        train_data, feature_columns
    )
    test_enhanced, _, _, _ = smart_ensemble.create_target_specific_features(test_data, feature_columns)

    print(f"基础特征数量: {len(base_features)}")
    print(f"信号特征数量: {len(signal_features)}")
    print(f"总特征数量: {len(all_features)}")

    # 准备数据
    X_train = train_enhanced[all_features]
    X_test = test_enhanced[all_features]
    y_train = train_data[target_columns].values
    y_test = test_data[target_columns].values

    # 划分验证集（用于模型选择）
    split_idx = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    print(f"训练数据: X_train{X_tr.shape}, y_train{y_tr.shape}")
    print(f"验证数据: X_val{X_val.shape}, y_val{y_val.shape}")
    print(f"测试数据: X_test{X_test.shape}, y_test{y_test.shape}")

    # 模型选择
    print("3. 为每个目标变量选择最佳模型...")
    validation_results = smart_ensemble.select_best_model_per_target(
        X_tr, y_tr, X_val, y_val, target_columns
    )

    # 使用完整数据训练最终模型
    print("4. 使用完整数据训练最终模型...")
    feature_sets = {
        'base_features': base_features,
        'signal_features': signal_features + base_features  # 信号特征包含基础特征
    }

    smart_ensemble.train_final_models(X_train, y_train, target_columns, feature_sets)

    # 预测
    print("5. 预测...")
    predictions = smart_ensemble.predict_with_best_models(X_test, target_columns, feature_sets)

    # 评估
    print("6. 评估...")
    results, overall_mae = smart_ensemble.evaluate_performance(y_test, predictions, target_columns)

    # 保存结果
    print("7. 保存结果...")
    results_df = []
    for i in tqdm(range(len(y_test)), desc="保存结果"):
        true_str = ' '.join(map(str, y_test[i]))
        pred_str = ' '.join(map(str, predictions[i]))
        error_str = ' '.join(map(str, np.abs(y_test[i] - predictions[i])))
        results_df.append([true_str, pred_str, error_str])

    result_df = pd.DataFrame(results_df, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("result_Smart_Model_Selection.csv", index=False)

    # 计算平均误差
    errors = np.abs(y_test - predictions)
    mean_errors = np.mean(errors, axis=0)

    print("\n" + "=" * 50)
    print("最终平均误差")
    print("=" * 50)
    for i, col in enumerate(target_columns):
        print(f"{col}: {mean_errors[i]:.6f}")
    print(f"总体平均误差: {overall_mae:.6f}")

    # 与目标对比
    target_error = 0.2
    print(f"\n目标误差: {target_error}")
    print(f"当前误差: {overall_mae:.6f}")
    print(f"差距: {overall_mae - target_error:.6f}")

    if overall_mae <= target_error:
        print(" 已达到目标误差!")
    else:
        improvement_needed = ((overall_mae - target_error) / target_error) * 100
        print(f"📈 还需要改善 {improvement_needed:.1f}%")

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\n总运行时间: {total_time:.1f} 分钟")


if __name__ == "__main__":
    main()
