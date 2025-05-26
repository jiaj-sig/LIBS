# LIBS等离子体诊断：温度和电子密度计算方法

## 目录
1. [引言](#引言)
2. [等离子体温度计算](#等离子体温度计算)
3. [电子密度计算](#电子密度计算)
4. [实际计算示例](#实际计算示例)
5. [注意事项和误差分析](#注意事项和误差分析)
6. [实验条件对测量的影响](#实验条件对测量的影响)
7. [结论](#结论)

---

## 引言

激光诱导击穿光谱（LIBS）技术中，等离子体的温度和电子密度是两个关键的物理参数。这些参数不仅决定了等离子体的物理化学性质，还直接影响光谱线的强度、展宽和形状，进而影响定量分析的准确性。本文档详细介绍了LIBS中等离子体温度和电子密度的主要计算方法。

---

## 等离子体温度计算

### 1. Boltzmann图法

#### 1.1 理论基础

在局部热力学平衡（LTE）条件下，原子在不同能级上的布居遵循Boltzmann分布：

$$\frac{N_i}{N_0} = \frac{g_i}{g_0} \exp\left(-\frac{E_i}{kT}\right)$$

其中：
- $N_i$：第$i$能级的原子数密度
- $N_0$：基态原子数密度  
- $g_i$：第$i$能级的统计权重
- $E_i$：第$i$能级的能量
- $k$：Boltzmann常数
- $T$：等离子体温度

#### 1.2 谱线强度关系

谱线强度与能级布居的关系为：

$$I_{ij} = \frac{hc}{4\pi\lambda_{ij}} A_{ij} N_i$$

其中：
- $I_{ij}$：从能级$i$到能级$j$的谱线强度
- $h$：Planck常数
- $c$：光速
- $\lambda_{ij}$：跃迁波长
- $A_{ij}$：Einstein自发发射系数

#### 1.3 Boltzmann图方程

结合上述两个方程，可得到：

$$\ln\left(\frac{I\lambda}{gA}\right) = \ln\left(\frac{hcN_0}{4\pi}\right) - \frac{E_i}{kT}$$

**实际应用中的关键公式：**

$$\boxed{\ln\left(\frac{I\lambda}{gA}\right) = -\frac{5040 \times E_i}{T} + C}$$

其中$E_i$的单位为eV，$T$的单位为K。

#### 1.4 计算步骤

1. 选择同一元素同一电离态的多条谱线（至少3-4条）
2. 测量各谱线的积分强度$I$
3. 查找对应的波长$\lambda$、统计权重$g$、跃迁概率$A$和上能级能量$E_i$
4. 计算$\ln(I\lambda/gA)$
5. 以$E_i$为横坐标，$\ln(I\lambda/gA)$为纵坐标作线性拟合
6. 从直线斜率$m = -5040/T$求得温度$T$

### 2. 双线法

当只有两条谱线可用时，可使用双线法：

$$\boxed{T = \frac{5040 \times (E_2 - E_1)}{\ln\left[\frac{I_1\lambda_1g_2A_2}{I_2\lambda_2g_1A_1}\right]}}$$

### 3. Saha-Boltzmann图法

对于同时包含原子线和离子线的情况：

$$\ln\left(\frac{I_{\text{ion}}\lambda_{\text{ion}}}{I_{\text{atom}}\lambda_{\text{atom}}} \times \frac{g_{\text{atom}}A_{\text{atom}}}{g_{\text{ion}}A_{\text{ion}}}\right) = -\frac{5040(E_{\text{ion}} + E_{\text{ionization}} - E_{\text{atom}})}{T} + \ln\left(\frac{2U_{\text{ion}}}{U_{\text{atom}}}\right) - \ln(N_e) + 15.68$$

---

## 电子密度计算

### 1. Stark展宽法

#### 1.1 理论基础

在等离子体中，谱线的Stark展宽主要由电子碰撞引起，展宽程度与电子密度成正比。

#### 1.2 基本公式

完整的Stark展宽公式为：

$$\Delta\lambda_{1/2} = 2w \times \left(\frac{N_e}{10^{16}}\right) \times \left[1 + 1.75A\left(\frac{N_e}{10^{16}}\right)^{1/4} \times (1-0.75R)\right]$$

其中：
- $\Delta\lambda_{1/2}$：谱线半高全宽（FWHM）
- $w$：Stark展宽参数
- $N_e$：电子密度（cm⁻³）
- $A$：离子展宽参数
- $R$：离子-电子密度比

#### 1.3 简化公式

在低密度近似下（$N_e < 10^{17}$ cm⁻³）：

$$\boxed{N_e = \frac{\Delta\lambda_{1/2}}{2w} \times 10^{16}}$$

#### 1.4 常用谱线的Stark展宽参数

**氢原子H_α线（656.3 nm）：**

$$w = 0.548 \text{ Å (at } T = 10000 \text{ K)}$$

$$\boxed{N_e = 8.02 \times 10^{12} \times (\Delta\lambda_{1/2})^{1.46} \text{ cm}^{-3}}$$

**氢原子H_β线（486.1 nm）：**

$$w = 0.319 \text{ Å (at } T = 10000 \text{ K)}$$

$$\boxed{N_e = 1.26 \times 10^{13} \times (\Delta\lambda_{1/2})^{1.46} \text{ cm}^{-3}}$$

### 2. Saha方程法

#### 2.1 基本公式

Saha方程描述了电离平衡：

$$\frac{N_{\text{ion}} \cdot N_e}{N_{\text{atom}}} = \frac{2U_{\text{ion}}}{U_{\text{atom}}} \left(\frac{2\pi m_e kT}{h^2}\right)^{3/2} \exp\left(-\frac{E_{\text{ionization}}}{kT}\right)$$

#### 2.2 实用形式

$$\boxed{\log(N_e) = \log\left(\frac{N_{\text{ion}}}{N_{\text{atom}}}\right) + \log\left(\frac{2U_{\text{ion}}}{U_{\text{atom}}}\right) + 15.68 - \frac{5040 \times E_{\text{ionization}}}{T}}$$

其中：
- $U_{\text{ion}}, U_{\text{atom}}$：离子和原子的配分函数
- $E_{\text{ionization}}$：电离能（eV）
- $m_e$：电子质量

### 3. 连续辐射法

连续辐射强度与电子密度的平方成正比：

$$I_{\text{continuum}} = C \times N_e^2 \times \exp\left(-\frac{h\nu}{kT}\right)$$

因此：

$$\boxed{N_e = \sqrt{\frac{I_{\text{continuum}} \times \exp(h\nu/kT)}{C}}}$$

---

## 实际计算示例

### 1. 温度计算示例

假设测得Fe原子的几条谱线数据：

| 波长 (Å) | 强度 (a.u.) | g | A (s⁻¹) | E_i (eV) |
|----------|-------------|---|---------|----------|
| 5269.5   | 1000        | 7 | 6.4×10⁷ | 4.28     |
| 5328.0   | 800         | 5 | 5.1×10⁷ | 4.26     |
| 5371.5   | 600         | 9 | 7.2×10⁷ | 4.30     |

**计算步骤：**

1. 计算$\ln(I\lambda/gA)$值：

   - Line 1: $\ln\left(\frac{1000 \times 5269.5}{7 \times 6.4 \times 10^7}\right) = \ln(11.76) = 2.46$
   
   - Line 2: $\ln\left(\frac{800 \times 5328.0}{5 \times 5.1 \times 10^7}\right) = \ln(16.70) = 2.81$
   
   - Line 3: $\ln\left(\frac{600 \times 5371.5}{9 \times 7.2 \times 10^7}\right) = \ln(4.97) = 1.60$

2. 进行线性拟合，从斜率求得温度。

### 2. 电子密度计算示例

测得H_α线的半高全宽为0.8 Å：

$$N_e = 8.02 \times 10^{12} \times (0.8)^{1.46} = 5.8 \times 10^{15} \text{ cm}^{-3}$$

---

## 注意事项和误差分析

### 1. 温度计算的注意事项

#### 1.1 选线原则
- 选择同一元素同一电离态的谱线
- 避免自吸收严重的谱线
- 上能级能量差应足够大（> 1 eV）
- 谱线应无重叠，强度适中

#### 1.2 主要误差来源
- 光谱仪分辨率和波长精度限制
- 原子数据（A值、g值）的不确定性
- 非热力学平衡效应
- 自吸收和自发射效应
- 基体效应和光谱干扰

### 2. 电子密度计算的注意事项

#### 2.1 Stark展宽法的限制
- 需要足够高的光谱分辨率（< 0.1 Å）
- 必须扣除仪器展宽的影响
- Stark展宽参数的温度依赖性
- 适用的电子密度范围：10¹⁴ - 10¹⁸ cm⁻³

#### 2.2 仪器展宽修正

实际的Stark展宽需要扣除仪器展宽：

$$\boxed{\Delta\lambda_{\text{Stark}} = \sqrt{\Delta\lambda_{\text{measured}}^2 - \Delta\lambda_{\text{instrument}}^2}}$$

#### 2.3 温度修正

Stark展宽参数的温度依赖性：

$$\boxed{w(T) = w_{\text{ref}} \times \left(\frac{T}{T_{\text{ref}}}\right)^\alpha}$$

其中α通常为0.5-1.0。

---

## 实验条件对测量的影响

### 1. 激光参数的影响

- **激光能量**：影响等离子体温度和密度
- **脉冲宽度**：影响等离子体的形成和演化
- **聚焦条件**：影响功率密度和等离子体特性

### 2. 检测参数的影响

- **延迟时间**：影响等离子体的演化阶段
- **门宽**：影响信号积分和时间分辨率
- **光谱分辨率**：影响谱线展宽的准确测量

---

## 结论

LIBS等离子体的温度和电子密度是表征等离子体状态的重要参数。Boltzmann图法是测量温度的标准方法，而Stark展宽法是测量电子密度的主要手段。在实际应用中，需要根据具体的实验条件和要求选择合适的计算方法，并注意各种误差来源的影响。

准确的等离子体诊断不仅有助于理解LIBS的物理机制，还为优化实验条件、提高分析精度提供了重要依据。随着LIBS技术在工业应用中的推广，等离子体诊断技术的重要性将日益凸显。

---

## 参考文献

1. Cremers, D. A., & Radziemski, L. J. (2013). *Handbook of laser-induced breakdown spectroscopy*. John Wiley & Sons.
2. Miziolek, A. W., Palleschi, V., & Schechter, I. (Eds.). (2006). *Laser-induced breakdown spectroscopy (LIBS): fundamentals and applications*. Cambridge University Press.
3. Griem, H. R. (1997). *Principles of plasma spectroscopy*. Cambridge University Press.
4. NIST Atomic Spectra Database. https://www.nist.gov/pml/atomic-spectra-database

---

*文档创建日期：2024年*  
*适用于：LIBS技术研究与应用* 