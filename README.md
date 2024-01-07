# 编程基础期末大作业

## 项目简介
这个项目包含一个ipynb文件，用于执行数据处理和机器学习分析。主要包括数据清洗、特征选择、逻辑回归、支持向量机（SVM）、随机森林等机器学习方法的应用。旨在通过实际数据集展示数据预处理、模型训练和评估的典型流程。

## 作者信息
中国人民大学  &nbsp;&nbsp;&nbsp; MyZhang

邮箱：zhangmy1366@ruc.edu.cn

## 运行环境
- Python 3.11
- Jupyter Notebook (推荐用于运行和展示代码)

## 使用的主要包
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- plotnine

## 安装依赖
在Python环境中运行以下命令以安装必要的包：
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib plotnine
```
以上命令用于安装Python包(默认最新版本即可)。

## 程序说明
程序主要含有两个类data_process 和 machine_learning，分别实现以下功能：

### data_process 类

#### 概述

`data_process` 类用于处理数据文件，包括读取不同格式的文件，处理离散和连续变量，以及保存处理后的数据。

#### 构造函数

`__init__(self, name, style)`

- `name` (str): 文件名称，不包括文件扩展名。
- `style` (str): 文件格式，支持 'csv'、'xlsx' 和 'xls'。

#### 方法

`read_file(self)`

- 从文件中读取数据，支持 'csv'、'xlsx' 和 'xls' 格式。
- 将读取的数据存储在 `df` 属性中。
- 将 '违约' 列的值映射为 0 和 1。

`discrete_variance(self, df, col_name)`

- 将离散变量转换为虚拟/指标变量。
- 返回包含虚拟变量的 DataFrame。

`continuous_variance(self, df, col_name)`

- 标准化连续变量。
- 返回标准化后的 DataFrame。

`main(self)`

- 主要处理数据的方法，依次执行以下步骤：
  1. 调用 `read_file` 方法读取数据文件。
  2. 处理离散变量，将其转换为虚拟变量。
  3. 处理连续变量，标准化数据。
  4. 将处理后的数据保存为 '处理后的数据.csv' 文件。
  5. 返回处理后的数据 DataFrame。

#### 异常处理

- 如果在构造函数中指定的文件格式不是支持的格式（'csv'、'xlsx' 或 'xls'），将引发 `ValueError` 异常。

#### 使用示例

```python
# 创建 data_process 实例
data_processor = data_process('data_file', 'csv')

# 执行数据处理
processed_data = data_processor.main()

# 查看处理后的数据
print(processed_data)
```

### machine_learning 类

#### 概述

`machine_learning` 类提供了一组用于机器学习任务的方法，包括数据分割、特征选择、逻辑回归、正则化逻辑回归、支持向量机和随机森林等。该类旨在帮助用户快速构建、训练和评估机器学习模型。

#### 构造函数

`__init__(self, df, target_column)`

- `df` (pd.DataFrame): 包含数据的 DataFrame。
- `target_column` (str): 目标列的名称，用于分类问题的目标。

#### 方法

`split_data(self, test_size=0.2, random_state=None)`

- 从数据中生成训练集和测试集。
- 可以指定测试集的比例和随机种子。
- 返回包含训练集和测试集的元组。

`stepwise_selection(self, verbose=True)`

- 执行基于AIC（Akaike信息准则）的前向-后向特征选择。
- 可以选择是否打印包含和排除的顺序。
- 返回选择的特征列表。

`feature_selection(self)`

- 调用 `stepwise_selection` 方法进行特征选择。
- 打印选定的特征列表。
- 返回选择的特征列表。

`LogisticRegression(self, threshold=0.5, feature_selection=True)`

- 执行逻辑回归模型的训练和评估。
- 可以选择是否进行特征选择。
- 返回AUC和准确率。

`select_best_threshold(self)`

- 选择最佳的分类阈值，通过尝试不同的阈值来评估逻辑回归模型。
- 打印最佳阈值、对应的AUC和准确率。
- 返回最佳阈值。

`LogisticRegressionl1(self)`

- 执行正则化逻辑回归模型的训练和评估（L1正则化）。
- 打印最佳的正则化参数倒数和模型的性能指标。

`Support_Vector_Machine(self, kernel='linear')`

- 执行支持向量机模型的训练和评估。
- 可以选择使用不同的核函数。
- 打印模型的准确性和分类报告。

`Random_Forest(self)`

- 执行随机森林模型的训练和评估。
- 打印模型的准确性和分类报告。

#### 异常处理

- 如果构造函数中的 `target_column` 不在数据框中，将引发 `KeyError` 异常。

#### 使用示例

```python
# 创建 machine_learning 实例
ml = machine_learning(data, '违约')

# 执行逻辑回归模型
ml.LogisticRegression()

# 执行正则化逻辑回归模型
ml.LogisticRegressionl1()

# 执行支持向量机模型
ml.Support_Vector_Machine('lenear')

# 执行随机森林模型
ml.Random_Forest()
```
## 复现步骤
在jupyter中逐单元格运行即可