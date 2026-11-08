"""資料整合、特徵工程與模型前處理。"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'raw'
TARGET = '是否為VIP（目標變數(y)）'


def load_data():
    """維持原整合方式，以明細成交價計算折扣後金額。"""
    customers = pd.read_csv(DATA_DIR / 'customers.csv', parse_dates=['signup_date'])
    orders = pd.read_csv(DATA_DIR / 'orders.csv', parse_dates=['order_date'])
    items = pd.read_csv(DATA_DIR / 'order_items.csv')
    products = pd.read_csv(DATA_DIR / 'products.csv')
    for frame, key in [(customers, 'customer_id'), (orders, 'order_id'), (products, 'product_id')]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f'{key} 有缺值或重複主鍵')
    if not (orders.customer_id.isin(customers.customer_id).all() and items.order_id.isin(orders.order_id).all() and items.product_id.isin(products.product_id).all()):
        raise ValueError('資料表包含無法對應的外鍵')
    if items[['quantity', 'unit_price', 'discount_rate']].isna().any().any() or (items.quantity <= 0).any() or (items.unit_price < 0).any() or not items.discount_rate.between(0, 1).all():
        raise ValueError('商品明細含有無效數量、價格或折扣')
    items['amount'] = items.quantity * items.unit_price * (1 - items.discount_rate)
    sales = (items.merge(products[['product_id', 'category']], on='product_id', validate='many_to_one')
             .merge(orders, on='order_id', validate='many_to_one')
             .merge(customers, on='customer_id', validate='many_to_one'))
    totals = items.groupby('order_id').amount.sum().rename('order_amount')
    order_data = orders.merge(totals, on='order_id', validate='one_to_one').merge(customers, on='customer_id', validate='many_to_one')
    return customers, orders, sales, order_data


def classification_data(customers, order_data):
    completed = order_data[order_data.status.eq('completed')]
    reference = order_data.order_date.max() + pd.Timedelta(days=1)
    summary = completed.groupby('customer_id').agg(order_count=('order_id', 'count'), total_spend=('order_amount', 'sum'), average_spend=('order_amount', 'mean'), last_order=('order_date', 'max'))
    frame = customers.merge(summary, on='customer_id', how='left', validate='one_to_one')
    frame['recency_days'] = (reference - frame.last_order).dt.days
    frame['tenure_days'] = (reference - frame.signup_date).dt.days.clip(lower=0)
    for col in ['order_count', 'total_spend', 'average_spend']:
        frame[col] = frame[col].fillna(0)
    # 目標欄位僅供訓練，不可加入輸入特徵。
    frame[TARGET] = frame.segment.eq('vip').astype(int)
    features = ['acquisition_channel', 'city', 'order_count', 'total_spend', 'average_spend', 'recency_days', 'tenure_days']
    return frame, features


def regression_data(order_data):
    """同日訂單彼此不可見，歷史統計僅包含更早日期。"""
    frame = order_data[order_data.status.eq('completed')].copy()
    daily = frame.groupby(['customer_id', 'order_date']).agg(day_count=('order_id', 'count'), day_spend=('order_amount', 'sum')).reset_index().sort_values(['customer_id', 'order_date'])
    daily['prior_orders'] = daily.groupby('customer_id').day_count.cumsum() - daily.day_count
    daily['prior_spend'] = daily.groupby('customer_id').day_spend.cumsum() - daily.day_spend
    frame = frame.merge(daily[['customer_id', 'order_date', 'prior_orders', 'prior_spend']], on=['customer_id', 'order_date'], validate='many_to_one')
    frame['prior_average'] = frame.prior_spend / frame.prior_orders.replace(0, np.nan)
    frame['tenure_days'] = (frame.order_date - frame.signup_date).dt.days.clip(lower=0)
    frame['month'] = frame.order_date.dt.month
    frame['weekday'] = frame.order_date.dt.dayofweek
    features = ['acquisition_channel', 'city', 'payment_type', 'prior_orders', 'prior_spend', 'prior_average', 'tenure_days', 'month', 'weekday']
    return frame.sort_values(['order_date', 'order_id']), features


def pipeline_for(X, estimator):
    """將缺值處理、標準化、類別編碼和模型一起匯出。"""
    categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', estimator)])
