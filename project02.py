"""執行方式：streamlit run project02.py。"""
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib
from project02_core import load_data

st.set_page_config(page_title='客戶與訂單預測', layout='wide')
st.title('客戶與訂單預測專案')

@st.cache_data
def get_data():
    return load_data()

try:
    customers, orders, sales, order_data = get_data()
except (OSError, ValueError, KeyError) as exc:
    st.error(f'資料載入失敗：{exc}')
    st.stop()

page = st.sidebar.radio('頁面', ['資料概覽', 'VIP 分類', '訂單金額預測'])
if st.sidebar.button('重新載入資料'):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

if page == '資料概覽':
    st.header('資料狀態與視覺化')
    a, b, c = st.columns(3)
    a.metric('客戶數', len(customers))
    b.metric('訂單數', len(orders))
    c.metric('整合明細數', len(sales))
    st.caption('金額＝數量 × 明細單價 ×（1 − 折扣率）；營收僅統計已完成訂單。')
    st.subheader('缺失值統計')
    st.dataframe(sales.isna().sum().rename('缺失數').to_frame(), use_container_width=True)
    completed = sales[sales.status.eq('completed')].copy()
    st.subheader('每月營收')
    completed['month'] = completed.order_date.dt.to_period('M').astype(str)
    st.line_chart(completed.groupby('month').amount.sum())
    left, right = st.columns(2)
    with left:
        st.subheader('商品類別營收')
        st.bar_chart(completed.groupby('category').amount.sum())
    with right:
        st.subheader('客戶分群人數')
        st.bar_chart(customers.segment.value_counts())
    st.subheader('整合資料預覽（前 500 筆）')
    st.dataframe(sales.head(500), use_container_width=True)
    st.download_button('下載整合資料 CSV', sales.to_csv(index=False).encode('utf-8-sig'), 'integrated_sales.csv', 'text/csv')
else:
    classification = page == 'VIP 分類'
    st.header(page)
    # 只讀取已訓練的完整 Pipeline，不在頁面執行訓練。
    files = {
        '羅吉斯迴歸': 'vip_logistic.joblib',
        '隨機森林': 'vip_random_forest.joblib' if classification else 'amount_random_forest.joblib',
        'XGBoost': 'amount_xgboost.joblib',
    }
    choices = ['羅吉斯迴歸', '隨機森林'] if classification else ['隨機森林', 'XGBoost']
    name = st.selectbox('模型', choices)
    model_path = Path(__file__).resolve().parent / 'models' / files[name]

    @st.cache_resource
    def get_model(path, modified_time):
        return joblib.load(path)

    try:
        bundle = get_model(str(model_path), model_path.stat().st_mtime_ns)
    except (OSError, ValueError, ImportError, EOFError) as exc:
        st.error(f'無法載入模型：{exc}')
        st.info('請先執行：python train_project02.py')
        st.stop()

    if classification:
        st.caption('依輸入資料預測是否具備 VIP 特徵。標籤來自目前客群，結果不代表未來一定升級。')
        with st.expander('檢視分類資料與目標變數(y)'):
            st.caption('是否為VIP（目標變數(y)）：1＝VIP，0＝非 VIP；此欄不作為輸入特徵。')
            preview = customers.copy()
            preview['是否為VIP（目標變數(y)）'] = preview.segment.eq('vip').astype(int)
            st.dataframe(preview, use_container_width=True)
    st.subheader('輸入資料')
    st.caption('填妥資料後，請按「開始預測」。消費統計請使用預測當下已知的資料。')
    labels = {
        'acquisition_channel': '客戶來源管道', 'city': '居住城市',
        'payment_type': '付款方式', 'order_count': '已完成訂單數',
        'total_spend': '累計消費金額', 'average_spend': '平均消費金額',
        'recency_days': '距離最近一次消費天數', 'tenure_days': '加入會員天數',
        'prior_orders': '歷史已完成訂單數', 'prior_spend': '歷史累計消費金額',
        'prior_average': '歷史平均訂單金額', 'month': '訂單月份', 'weekday': '訂單星期',
    }
    translations = {
        'ads': '廣告', 'social': '社群媒體', 'organic': '自然搜尋',
        'referral': '推薦', 'email': '電子郵件', 'direct': '直接造訪',
        'Taipei': '台北', 'NewTaipei': '新北', 'New Taipei': '新北',
        'Taoyuan': '桃園', 'Taichung': '台中', 'Tainan': '台南', 'Kaohsiung': '高雄',
        'atm': 'ATM 轉帳', 'cod': '貨到付款', 'wallet': '電子錢包',
        'credit_card': '信用卡', 'card': '信用卡',
    }
    values = {}
    integer_fields = {'order_count', 'prior_orders', 'recency_days', 'tenure_days'}
    for feature in bundle['features']:
        setting = bundle['inputs'][feature]
        widget_key = f'{page}_{feature}'
        if feature == 'month':
            values[feature] = st.selectbox(labels[feature], list(range(1, 13)), index=int(setting['default']) - 1, key=widget_key)
        elif feature == 'weekday':
            days = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
            values[feature] = st.selectbox(labels[feature], list(range(7)), index=int(setting['default']), format_func=lambda value: days[value], key=widget_key)
        elif 'options' in setting:
            values[feature] = st.selectbox(labels[feature], setting['options'], format_func=lambda value: translations.get(value, value), key=widget_key)
        elif feature in integer_fields:
            values[feature] = st.number_input(labels[feature], min_value=0, value=max(0, int(setting['default'])), step=1, key=widget_key)
        else:
            values[feature] = st.number_input(labels[feature], min_value=0.0, value=max(0.0, float(setting['default'])), key=widget_key)
    # 只有使用者按下按鈕，才呼叫模型並顯示本次輸入的預測結果。
    if not st.button('開始預測', type='primary', key=f'{page}_predict'):
        st.stop()
    row = pd.DataFrame([values])[bundle['features']]
    model = bundle['model']
    st.subheader('預測結果')
    if classification:
        predicted = int(model.predict(row)[0])
        st.metric('VIP 資格預測', '具備 VIP 特徵' if predicted == 1 else '未具備 VIP 特徵')
        vip_index = list(model.classes_).index(1)
        st.metric('VIP 預測機率', f'{model.predict_proba(row)[0, vip_index]:.1%}')
    else:
        st.metric('預測訂單金額', f'{max(0, float(model.predict(row)[0])):,.2f}')
