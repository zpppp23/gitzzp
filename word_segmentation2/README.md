# 邮件分类系统

## 核心功能说明
基于多项式朴素贝叶斯算法实现的中文邮件分类系统，支持高频词和TF-IDF两种特征提取模式。

## 算法基础
### 多项式朴素贝叶斯分类器
采用基于条件概率的特征独立性假设
采用**特征条件独立性假设**：
```math
P(y|X) \propto P(y) \prod_{i=1}^n P(x_i|y)
```
### 贝叶斯定理应用公式
先验概率计算：
P(y=c) = \frac{N_c + \alpha}{N + \alpha \times C}

N_c：类别c的样本数

α：拉普拉斯平滑系数

条件概率计算：
P(x_i|y=c) = \frac{N_{x_i,c} + \alpha}{N_c + \alpha \times V}

V：特征词表大小
- **贝叶斯定理应用**：

  - 先验概率：`P(y)` 通过类别频率计算
  - 似然概率：`P(xᵢ|y)` 使用平滑后的词频统计
  - 决策规则：`argmax_y P(y)∏ᵢP(xᵢ|y)`

### 特征构建对比
高频词特征（HFC）
数学表达：
```math
\mathbf{x}_i = \begin{bmatrix} 
I(w_1 \in d_i) \\ 
\vdots \\ 
I(w_K \in d_i)
\end{bmatrix}
```
实现优势：
```python
CountVectorizer(max_features=1000,binary=True) # 启用二进制模式
```
TF-IDF特征
数学表达：

```math
\text{TF-IDF}(t,d) = \underbrace{\frac{f_{t,d}}{\sum_{t'}f_{t',d}}}_{\text{归一化词频}} \times \underbrace{\log\frac{N}{1 + n_t}}_{\text{逆文档频率}}
```
实现关键：
```python
TfidfVectorizer(max_features=1000,norm='l2')# 欧式归一化
```
## 特征模式切换
高频词模式：Countvectorizer,TF-IDF模式：Tfidfvectorizer
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vectorizer(mode):
    return {
        'hfc': CountVectorizer(max_features=1000, binary=True),
        'tfidf': TfidfVectorizer(max_features=1000)
    }[mode]
```
## 性能对比
| 特征类型   | 准确率   | 训练速度 | 适用场景 |
|--------|-------|------|------|
| HFC    | 82.3% | 1.2x | 实时系统 |
| TF-IDF | 85.7% | 1.0x | 精准分类 |

## 数据处理流程
```python
1. 中文分词
   - 使用jieba分词器：jieba.cut(content)
   
2. 停用词过滤
   - 加载停用词表：stopwords = [line.strip() for line in open('stopwords.txt')]
   - 过滤策略：token not in stopwords and len(token) > 1

3. 文本清洗
   - 正则表达式去除标点：re.sub(r'[^\w\s]', '', text)



   
