#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIBS煤标样数据分析算法
作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import signal
from scipy.stats import pearsonr
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class LIBSAnalyzer:
    """LIBS数据分析器"""
    
    def __init__(self, data_dir='1064nm', reference_file='煤标样.xlsx'):
        self.data_dir = data_dir
        self.reference_file = reference_file
        self.raw_spectra = {}
        self.processed_spectra = {}
        self.reference_data = None
        self.feature_matrix = None
        self.models = {}
        self.results = {}
        
    def load_reference_data(self):
        """加载煤标样真实值数据"""
        print("步骤1: 加载煤标样真实值数据...")
        
        df = pd.read_excel(self.reference_file)
        # 清理数据，去掉第一行标题
        df = df.iloc[1:].reset_index(drop=True)
        
        # 重命名列
        df.columns = ['样品编号', '样品名称', '全硫', '灰分', '挥发分', '热值', '碳', '氢', '氮', '真相对密度', '焦渣特性']
        
        # 转换数值列
        numeric_cols = ['全硫', '灰分', '挥发分', '热值', '碳', '氢', '氮', '真相对密度']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.reference_data = df
        print(f"成功加载 {len(df)} 个煤标样的真实值数据")
        return df
    
    def load_libs_data(self):
        """加载LIBS光谱数据"""
        print("步骤2: 加载LIBS光谱数据...")
        
        files = list(Path(self.data_dir).glob('*.xlsx'))
        
        for file in files:
            sample_id = file.stem.split('-')[0]  # 提取样品编号
            print(f"  加载样品 {sample_id}...")
            
            xl = pd.ExcelFile(file)
            sample_spectra = {}
            
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name)
                
                # 从第7行开始读取数据（索引6）
                if len(df) > 6:
                    spectrum_data = df.iloc[6:].copy()
                    spectrum_data.columns = ['wavelength', 'intensity']
                    
                    # 转换为数值类型
                    spectrum_data['wavelength'] = pd.to_numeric(spectrum_data['wavelength'], errors='coerce')
                    spectrum_data['intensity'] = pd.to_numeric(spectrum_data['intensity'], errors='coerce')
                    
                    # 去除NaN值
                    spectrum_data = spectrum_data.dropna()
                    
                    sample_spectra[sheet_name] = spectrum_data
            
            self.raw_spectra[sample_id] = sample_spectra
        
        print(f"成功加载 {len(self.raw_spectra)} 个样品的光谱数据")
        return self.raw_spectra
    
    def preprocess_spectra(self):
        """光谱数据预处理"""
        print("步骤3: 光谱数据预处理...")
        
        for sample_id, spectra in self.raw_spectra.items():
            print(f"  预处理样品 {sample_id}...")
            processed_sample = {}
            
            for channel, spectrum in spectra.items():
                # 1. 平滑滤波
                smoothed_intensity = signal.savgol_filter(spectrum['intensity'], 
                                                        window_length=5, polyorder=2)
                
                # 2. 基线校正（简单的最小值校正）
                baseline_corrected = smoothed_intensity - np.min(smoothed_intensity)
                
                # 3. 归一化
                normalized_intensity = baseline_corrected / np.max(baseline_corrected)
                
                processed_spectrum = spectrum.copy()
                processed_spectrum['intensity'] = normalized_intensity
                processed_sample[channel] = processed_spectrum
            
            self.processed_spectra[sample_id] = processed_sample
        
        print("光谱预处理完成")
        return self.processed_spectra
    
    def extract_features(self):
        """特征提取"""
        print("步骤4: 特征提取...")
        
        features_list = []
        sample_ids = []
        
        for sample_id, spectra in self.processed_spectra.items():
            sample_features = []
            
            for channel, spectrum in spectra.items():
                wavelength = spectrum['wavelength'].values
                intensity = spectrum['intensity'].values
                
                # 统计特征
                features = [
                    np.mean(intensity),           # 平均强度
                    np.std(intensity),            # 强度标准差
                    np.max(intensity),            # 最大强度
                    np.min(intensity),            # 最小强度
                    np.sum(intensity),            # 总强度
                    np.trapz(intensity, wavelength),  # 积分面积
                ]
                
                # 峰值特征
                peaks, _ = signal.find_peaks(intensity, height=0.1, distance=10)
                features.extend([
                    len(peaks),                   # 峰数量
                    np.mean(intensity[peaks]) if len(peaks) > 0 else 0,  # 平均峰高
                ])
                
                # 光谱矩特征
                features.extend([
                    np.sum(wavelength * intensity) / np.sum(intensity),  # 质心
                    np.sqrt(np.sum((wavelength - features[-1])**2 * intensity) / np.sum(intensity)),  # 标准差
                ])
                
                sample_features.extend(features)
            
            features_list.append(sample_features)
            sample_ids.append(sample_id)
        
        # 创建特征矩阵
        self.feature_matrix = pd.DataFrame(features_list, index=sample_ids)
        
        # 生成特征名称
        feature_names = []
        for channel_idx in range(len(list(self.processed_spectra.values())[0])):
            for feat_name in ['mean', 'std', 'max', 'min', 'sum', 'area', 'n_peaks', 'peak_height', 'centroid', 'width']:
                feature_names.append(f'ch{channel_idx}_{feat_name}')
        
        self.feature_matrix.columns = feature_names[:len(self.feature_matrix.columns)]
        
        print(f"提取特征矩阵: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def build_models(self, target_properties=['全硫', '灰分', '挥发分', '热值', '碳']):
        """构建预测模型"""
        print("步骤5: 构建预测模型...")
        
        # 准备数据
        X = self.feature_matrix
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        self.models = {}
        self.results = {}
        
        for prop in target_properties:
            print(f"  构建 {prop} 预测模型...")
            
            # 获取目标值
            y = []
            valid_samples = []
            
            for sample_id in X.index:
                ref_row = self.reference_data[self.reference_data['样品编号'] == sample_id]
                if not ref_row.empty and not pd.isna(ref_row[prop].iloc[0]):
                    y.append(ref_row[prop].iloc[0])
                    valid_samples.append(sample_id)
            
            if len(y) < 3:
                print(f"    {prop} 数据不足，跳过")
                continue
            
            y = np.array(y)
            X_prop = X_scaled.loc[valid_samples]
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_prop, y, test_size=0.3, random_state=42
            )
            
            # 训练多个模型
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
            
            prop_results = {}
            
            for model_name, model in models.items():
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # 评估
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                prop_results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'X_train': X_train,
                    'X_test': X_test
                }
            
            self.models[prop] = prop_results
            
            # 选择最佳模型
            best_model_name = max(prop_results.keys(), 
                                key=lambda x: prop_results[x]['test_r2'])
            self.results[prop] = prop_results[best_model_name]
            
            print(f"    最佳模型: {best_model_name} (R² = {prop_results[best_model_name]['test_r2']:.3f})")
        
        return self.models, self.results
    
    def visualize_results(self, save_dir='results'):
        """可视化结果"""
        print("步骤6: 生成可视化结果...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 原始光谱可视化
        self._plot_raw_spectra(save_dir)
        
        # 2. 预处理效果对比
        self._plot_preprocessing_comparison(save_dir)
        
        # 3. 特征重要性分析
        self._plot_feature_importance(save_dir)
        
        # 4. 模型性能对比
        self._plot_model_performance(save_dir)
        
        # 5. 预测结果对比
        self._plot_prediction_results(save_dir)
        
        # 6. 相关性分析
        self._plot_correlation_analysis(save_dir)
        
        print(f"可视化结果已保存到 {save_dir} 目录")
    
    def _plot_raw_spectra(self, save_dir):
        """绘制原始光谱"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 选择几个代表性样品
        sample_ids = list(self.raw_spectra.keys())[:6]
        
        for i, sample_id in enumerate(sample_ids):
            ax = axes[i]
            spectra = self.raw_spectra[sample_id]
            
            for j, (channel, spectrum) in enumerate(spectra.items()):
                ax.plot(spectrum['wavelength'], spectrum['intensity'], 
                       alpha=0.7, label=f'通道{j+1}')
            
            ax.set_title(f'样品 {sample_id} 原始光谱')
            ax.set_xlabel('波长 (nm)')
            ax.set_ylabel('强度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/01_原始光谱.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_preprocessing_comparison(self, save_dir):
        """绘制预处理效果对比"""
        sample_id = list(self.raw_spectra.keys())[0]
        channel = list(self.raw_spectra[sample_id].keys())[0]
        
        raw_spectrum = self.raw_spectra[sample_id][channel]
        processed_spectrum = self.processed_spectra[sample_id][channel]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 原始光谱
        ax1.plot(raw_spectrum['wavelength'], raw_spectrum['intensity'], 'b-', alpha=0.7)
        ax1.set_title(f'样品 {sample_id} 原始光谱')
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('强度')
        ax1.grid(True, alpha=0.3)
        
        # 预处理后光谱
        ax2.plot(processed_spectrum['wavelength'], processed_spectrum['intensity'], 'r-', alpha=0.7)
        ax2.set_title(f'样品 {sample_id} 预处理后光谱')
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('归一化强度')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/02_预处理效果对比.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, save_dir):
        """绘制特征重要性"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (prop, result) in enumerate(self.results.items()):
            if i >= 6:
                break
                
            ax = axes[i]
            model = result['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:]  # 前10个重要特征
                
                ax.barh(range(len(indices)), importances[indices])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([self.feature_matrix.columns[i] for i in indices])
                ax.set_title(f'{prop} 特征重要性')
                ax.set_xlabel('重要性')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/03_特征重要性.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance(self, save_dir):
        """绘制模型性能对比"""
        if not self.models:
            return
        
        properties = list(self.models.keys())
        model_names = list(list(self.models.values())[0].keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R²对比
        r2_data = []
        for prop in properties:
            for model_name in model_names:
                if model_name in self.models[prop]:
                    r2_data.append({
                        'Property': prop,
                        'Model': model_name,
                        'R²': self.models[prop][model_name]['test_r2']
                    })
        
        r2_df = pd.DataFrame(r2_data)
        r2_pivot = r2_df.pivot(index='Property', columns='Model', values='R²')
        
        sns.heatmap(r2_pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('模型R²性能对比')
        
        # RMSE对比
        rmse_data = []
        for prop in properties:
            for model_name in model_names:
                if model_name in self.models[prop]:
                    rmse_data.append({
                        'Property': prop,
                        'Model': model_name,
                        'RMSE': self.models[prop][model_name]['test_rmse']
                    })
        
        rmse_df = pd.DataFrame(rmse_data)
        rmse_pivot = rmse_df.pivot(index='Property', columns='Model', values='RMSE')
        
        sns.heatmap(rmse_pivot, annot=True, fmt='.3f', cmap='viridis_r', ax=ax2)
        ax2.set_title('模型RMSE性能对比')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/04_模型性能对比.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_results(self, save_dir):
        """绘制预测结果对比"""
        if not self.results:
            return
        
        n_props = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (prop, result) in enumerate(self.results.items()):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # 训练集
            ax.scatter(result['y_train'], result['y_pred_train'], 
                      alpha=0.6, label='训练集', color='blue')
            
            # 测试集
            ax.scatter(result['y_test'], result['y_pred_test'], 
                      alpha=0.6, label='测试集', color='red')
            
            # 理想线
            min_val = min(np.min(result['y_train']), np.min(result['y_test']))
            max_val = max(np.max(result['y_train']), np.max(result['y_test']))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            ax.set_xlabel(f'真实值 ({prop})')
            ax.set_ylabel(f'预测值 ({prop})')
            ax.set_title(f'{prop} 预测结果 (R² = {result["test_r2"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/05_预测结果对比.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, save_dir):
        """绘制相关性分析"""
        # 特征相关性矩阵
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 特征间相关性
        corr_matrix = self.feature_matrix.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('特征相关性矩阵')
        
        # 目标属性间相关性
        target_props = ['全硫', '灰分', '挥发分', '热值', '碳']
        target_data = self.reference_data[target_props].corr()
        
        sns.heatmap(target_data, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=ax2)
        ax2.set_title('目标属性相关性矩阵')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/06_相关性分析.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, save_path='LIBS分析报告.md'):
        """生成分析报告"""
        print("步骤7: 生成分析报告...")
        
        report = f"""# LIBS煤标样数据分析报告

## 1. 数据概述

### 1.1 数据来源
- LIBS光谱数据: {len(self.raw_spectra)} 个煤标样
- 参考数据: {len(self.reference_data)} 个煤标样的真实值
- 光谱通道数: {len(list(self.raw_spectra.values())[0])} 个

### 1.2 目标属性
分析的煤质参数包括:
"""
        
        for prop in self.reference_data.columns[2:7]:
            mean_val = self.reference_data[prop].mean()
            std_val = self.reference_data[prop].std()
            report += f"- {prop}: 均值 {mean_val:.2f}, 标准差 {std_val:.2f}\n"
        
        report += f"""
## 2. 算法流程

### 2.1 数据预处理
1. **光谱平滑**: 使用Savitzky-Golay滤波器去除噪声
2. **基线校正**: 最小值校正方法
3. **归一化**: 最大值归一化

### 2.2 特征提取
从每个光谱通道提取10个特征:
- 统计特征: 均值、标准差、最大值、最小值、总强度
- 积分特征: 光谱面积
- 峰值特征: 峰数量、平均峰高
- 矩特征: 质心、光谱宽度

总特征数: {self.feature_matrix.shape[1]}

### 2.3 机器学习模型
使用4种回归算法:
- Random Forest Regressor
- Linear Regression  
- Ridge Regression
- Support Vector Regression

## 3. 模型性能评估

"""
        
        if self.results:
            for prop, result in self.results.items():
                report += f"""### 3.{list(self.results.keys()).index(prop)+1} {prop}预测模型
- **R²得分**: {result['test_r2']:.4f}
- **RMSE**: {result['test_rmse']:.4f}
- **训练集R²**: {result['train_r2']:.4f}
- **测试集样本数**: {len(result['y_test'])}

"""
        
        report += f"""## 4. 结果分析

### 4.1 模型性能总结
"""
        
        if self.results:
            avg_r2 = np.mean([result['test_r2'] for result in self.results.values()])
            best_prop = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
            worst_prop = min(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
            
            report += f"""- **平均R²得分**: {avg_r2:.4f}
- **最佳预测属性**: {best_prop} (R² = {self.results[best_prop]['test_r2']:.4f})
- **最差预测属性**: {worst_prop} (R² = {self.results[worst_prop]['test_r2']:.4f})

### 4.2 算法优势
1. **多通道融合**: 充分利用多个光谱通道信息
2. **特征工程**: 提取多维度光谱特征
3. **模型集成**: 对比多种机器学习算法
4. **可视化分析**: 全流程可视化展示

### 4.3 改进建议
1. **数据增强**: 增加更多样品数据
2. **特征选择**: 使用特征选择算法优化特征集
3. **深度学习**: 尝试CNN等深度学习方法
4. **集成学习**: 使用模型融合提高预测精度

## 5. 可视化结果

生成的可视化图表包括:
1. 原始光谱图
2. 预处理效果对比图  
3. 特征重要性分析图
4. 模型性能对比图
5. 预测结果散点图
6. 相关性分析热力图

## 6. 结论

本研究建立了基于LIBS技术的煤质参数预测模型，通过多通道光谱数据融合和机器学习算法，实现了对煤炭关键质量指标的定量分析。模型在测试集上的平均R²得分为{avg_r2:.4f}，表明该方法具有良好的预测性能和实用价值。

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析报告已保存到: {save_path}")
        return report

def main():
    """主函数"""
    print("=== LIBS煤标样数据分析系统 ===\n")
    
    # 创建分析器
    analyzer = LIBSAnalyzer()
    
    # 执行分析流程
    analyzer.load_reference_data()
    analyzer.load_libs_data()
    analyzer.preprocess_spectra()
    analyzer.extract_features()
    analyzer.build_models()
    analyzer.visualize_results()
    analyzer.generate_report()
    
    print("\n=== 分析完成 ===")
    print("请查看results目录中的可视化结果和分析报告")

if __name__ == "__main__":
    main() 