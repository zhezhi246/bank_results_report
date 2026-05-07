import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('Agg')  # 后台出图
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans

# ==========================================
# 0. 全局样式与字体配置
# ==========================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 50)
    print(" 开始执行：银行客户流失数据挖掘全流程分析 ")
    print("=" * 50)

    report_file = 'bank_results_report.csv'
    with open(report_file, 'w', encoding='utf-8-sig') as f:
        f.write("银行客户流失数据挖掘实战 - 综合结果报告\n\n")

    # ==========================================
    # 1. 数据加载与清洗
    # ==========================================
    print("\n[1/4] 正在加载和清洗数据...")
    df = pd.read_csv('Churn_Modelling.csv')

    # 剔除对预测毫无意义的列（行号、用户ID、姓氏）
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True, errors='ignore')

    # 目标变量为 Exited (1=流失, 0=留存)
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # 自动对文本列（如 Geography, Gender）进行 One-Hot 编码
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f" -> 数据清洗完成！共有样本 {X_encoded.shape[0]} 个，特征 {X_encoded.shape[1]} 个。")

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # 2. 任务一：流失预测分析
    # ==========================================
    print("\n[2/4] 正在训练随机森林预测模型...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # 记录评估报告
    report_dict = classification_report(y_test, y_pred, target_names=['留存 (0)', '流失 (1)'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    with open(report_file, 'a', encoding='utf-8-sig') as f:
        f.write("=== 任务一：流失预测模型评估 ===\n")
    report_df.to_csv(report_file, mode='a', encoding='utf-8-sig')

    # 绘图一：混淆矩阵与ROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('任务一：银行客户流失预测模型评估', fontsize=16, fontweight='bold')

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['预测留存', '预测流失'], yticklabels=['实际留存', '实际流失'], annot_kws={"size": 14})
    axes[0].set_title('混淆矩阵', fontsize=14)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC面积 = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlabel('假阳性率 (FPR)')
    axes[1].set_ylabel('真阳性率 (TPR)')
    axes[1].set_title('ROC 性能曲线', fontsize=14)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('bank_task1_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> [图片] 图表一已保存")

    # ==========================================
    # 3. 任务二：关键特征挖掘
    # ==========================================
    print("\n[3/4] 正在提取关键流失驱动因素...")
    importances = rf_model.feature_importances_
    features = X_train.columns
    feature_df = pd.DataFrame({'特征': features, '重要性权重': importances})

    top_features = feature_df.sort_values(by='重要性权重', ascending=False).head(10)
    top_features['重要性权重'] = top_features['重要性权重'].round(3)

    with open(report_file, 'a', encoding='utf-8-sig') as f:
        f.write("\n=== 任务二：影响客户流失的核心特征排行 (Top 10) ===\n")
    top_features.to_csv(report_file, mode='a', index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性权重', y='特征', data=top_features, hue='特征', palette='flare', legend=False)
    plt.title('任务二：影响银行客户流失的核心特征排行', fontsize=16, fontweight='bold')
    plt.xlabel('特征重要性得分')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('bank_task2_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> [图片] 图表二已保存")

    # ==========================================
    # 4. 任务三：用户画像与价值分群
    # ==========================================
    print("\n[4/4] 正在进行无监督学习：银行客户群体划分...")

    # 选取银行业务典型的数值特征：年龄、账户余额、预估薪水
    cluster_features = ['Age', 'Balance', 'EstimatedSalary']
    X_cluster = df[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 翻译分群标签，赋予业务含义
    cluster_mapping = {
        0: "A群: 低余额年轻客群",
        1: "B群: 高余额高龄客群 (流失高危)",
        2: "C群: 高薪资稳健客群"
    }
    df['User_Persona'] = df['Cluster'].map(cluster_mapping)

    summary = df.groupby('User_Persona')[cluster_features].mean().round(3)
    summary['用户数'] = df.groupby('User_Persona').size()

    with open(report_file, 'a', encoding='utf-8-sig') as f:
        f.write("\n=== 任务三：3大用户画像群业务特征统计 ===\n")
    summary.to_csv(report_file, mode='a', encoding='utf-8-sig')

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Age', y='Balance', hue='User_Persona', data=df,
                    palette='Set1', alpha=0.6, s=50)

    plt.title('任务三：银行客户年龄与余额画像矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('年龄 (岁)', fontsize=12)
    plt.ylabel('账户余额 (美元)', fontsize=12)
    plt.legend(title='用户画像标签', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('bank_task3_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> [图片] 图表三已保存")

    print("\n" + "=" * 50)
    print(" 数据挖掘分析全部完成！结果已汇总至：bank_results_report.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()