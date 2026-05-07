import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 页面配置 (Streamlit UI)
# ==========================================
st.set_page_config(page_title="银行客户流失预警系统", layout="wide")

st.title("🛡️ 银行客户流失风险 - 实时预测系统")
st.markdown("---")


# ==========================================
# 2. 核心：加载数据并训练模型 (带缓存机制)
# ==========================================
@st.cache_resource
def train_real_model():
    # 加载真实数据
    df = pd.read_csv('Churn_Modelling.csv')
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True, errors='ignore')

    # 预处理
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # 记录特征名称，用于后续预测对齐
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    # 训练随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_encoded, y)

    return model, feature_names, X['Geography'].unique(), X['Gender'].unique()


# 初始化模型
with st.spinner("正在基于真实数据训练随机森林模型，请稍候..."):
    model, feature_names, geo_list, gender_list = train_real_model()

# ==========================================
# 3. 侧边栏：交互式输入参数
# ==========================================
st.sidebar.header("📊 输入客户特征参数")


def user_input_features():
    # 侧边栏滑块和下拉框
    age = st.sidebar.slider("客户年龄 (Age)", 18, 92, 40)
    balance = st.sidebar.slider("账户余额 (Balance)", 0.0, 250000.0, 50000.0)
    credit_score = st.sidebar.slider("信用评分 (Credit Score)", 350, 850, 600)
    tenure = st.sidebar.slider("在网时长 (Tenure)", 0, 10, 5)
    num_products = st.sidebar.selectbox("购买产品数量 (NumOfProducts)", [1, 2, 3, 4], index=0)
    is_active = st.sidebar.selectbox("是否为活跃会员 (IsActiveMember)", ["是", "否"], index=0)
    has_card = st.sidebar.selectbox("是否有信用卡 (HasCrCard)", ["是", "否"], index=0)
    salary = st.sidebar.slider("预估薪资 (EstimatedSalary)", 0.0, 200000.0, 100000.0)
    geo = st.sidebar.selectbox("地理位置 (Geography)", geo_list)
    gender = st.sidebar.selectbox("性别 (Gender)", gender_list)

    # 将输入转化为模型需要的 Dataframe 格式
    data = {
        'CreditScore': credit_score,
        'Geography': geo,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_card == "是" else 0,
        'IsActiveMember': 1 if is_active == "是" else 0,
        'EstimatedSalary': salary
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# ==========================================
# 4. 实时预测逻辑
# ==========================================
# 同步 One-Hot 编码格式
input_encoded = pd.get_dummies(input_df)
# 补齐训练时有但输入时缺失的 One-Hot 列 (例如不同的国家)
full_input = pd.DataFrame(columns=feature_names)
full_input = pd.concat([full_input, input_encoded], axis=0).fillna(0)
full_input = full_input[feature_names]  # 确保列顺序完全一致

# 执行预测
prediction_proba = model.predict_proba(full_input)[0][1]
prediction = 1 if prediction_proba > 0.5 else 0

# ==========================================
# 5. 结果展示
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💡 预测结果")
    if prediction == 1:
        st.error(f"该客户具有 **高流失风险**")
    else:
        st.success(f"该客户状态 **稳定**")

    st.metric(label="AI 预测流失概率", value=f"{prediction_proba * 100:.2f}%")

    # 风险进度条
    st.progress(prediction_proba)

with col2:
    st.subheader("📝 客户简画像")
    st.write(f"**年龄阶段:** {input_df['Age'][0]} 岁")
    st.write(f"**财务状况:** 余额 ${input_df['Balance'][0]:,.2f} | 薪资 ${input_df['EstimatedSalary'][0]:,.2f}")
    st.write(f"**产品深度:** 购买了 {input_df['NumOfProducts'][0]} 款银行产品")

    # 给出基于逻辑的预警提示
    if input_df['NumOfProducts'][0] >= 3:
        st.warning("⚠️ 警告：该客户购买产品过多，在此数据集中通常预示极高流失风险。")
    if input_df['Age'][0] > 50 and input_df['Balance'][0] > 100000:
        st.info("ℹ️ 提示：该客户属于高净值高龄群体，需重点关注其账户变动。")

st.markdown("---")
st.caption("注：此工具基于随机森林分类器实时计算。概率结果完全来源于您的 `Churn_Modelling.csv` 真实数据规律。")