

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import math
from pathlib import Path

# 获取项目根目录（假设 traffic/ 在根目录下）
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models' / 'saved_models'
DATA_DIR = BASE_DIR / 'data' / 'processed'

lstm_model = load_model(MODEL_DIR / "lstm_model.keras")
gru_model = load_model(MODEL_DIR / "gru_model.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "onehot_encoder.pkl")

# 全局缓存历史数据
_historical_data = None

def _load_historical_data():
    """加载历史数据用于查询（内部使用）"""
    global _historical_data
    if _historical_data is None:
        # 加载所有可用的历史数据（train + val + test）
        train_df = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['timestamp'])
        val_df = pd.read_csv(DATA_DIR / 'val.csv', parse_dates=['timestamp'])
        test_df = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['timestamp'])
        _historical_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return _historical_data

def get_past_4_flows(site_id, departure_time):
    """
    根据站点和出发时间查询过去4个时间步的流量
    参数:
        site_id: int, 站点编号
        departure_time: datetime, 出发时间
    返回:
        list of 4 floats, 过去4个时间步的流量 [t-4, t-3, t-2, t-1]
    """
    df = _load_historical_data()
    
    # 筛选该站点的数据
    site_data = df[df['site_number'] == site_id].copy()
    
    if site_data.empty:
        raise ValueError(f"站点 {site_id} 没有历史数据")
    
    # 按时间排序
    site_data = site_data.sort_values('timestamp').reset_index(drop=True)
    
    # 查找最接近 departure_time 的时间点
    time_diffs = abs(site_data['timestamp'] - pd.to_datetime(departure_time))
    closest_idx = time_diffs.idxmin()
    
    # 获取该时间点的过去4个流量
    row = site_data.iloc[closest_idx]
    past_4_flows = [
        float(row['flow_t_minus_4']),
        float(row['flow_t_minus_3']),
        float(row['flow_t_minus_2']),
        float(row['flow_t_minus_1'])
    ]
    
    return past_4_flows

def predict_travel_time_by_departure(edge_features, model_name='lstm'):
    """
    供 routing 层直接调用的接口 - 根据出发时间自动查询历史数据
    edge_features 必须包含:
        - from_site: int
        - to_site: int
        - distance_km: float
        - departure_time: datetime 或 str (ISO格式)
    返回: float 通行时间（分钟）
    """
    # 查询历史数据获取 past_4_flows
    past_4_flows = get_past_4_flows(
        edge_features['from_site'],
        edge_features['departure_time']
    )
    
    # 构造完整的 edge_features
    full_edge_features = {
        'from_site': edge_features['from_site'],
        'to_site': edge_features['to_site'],
        'distance_km': edge_features['distance_km'],
        'past_4_flows': past_4_flows
    }
    
    # 调用原有的预测函数
    return predict_travel_time(full_edge_features, model_name)


def predict_flow(site_id, past_4_flows, model_name='lstm'):
    """
    past_4_flows: list of 4 floats, 顺序 [t-4, t-3, t-2, t-1]
    """
    # 1. 站点 one-hot 编码 (保持不变)
    site_encoded = encoder.transform([[site_id]]).flatten()  
    
    # 2. 修改这里：将 past_4_flows 转换为 (4, 1) 的形状再进行标准化
    past_4_flows_np = np.array(past_4_flows).reshape(-1, 1) # 变成 4行1列
    past_4_flows_scaled = scaler.transform(past_4_flows_np).flatten() # 标准化后展平回长度为 4 的数组
    
    # 3. 构造输入序列 (保持不变)
    seq_len = 4
    X = []
    for i in range(seq_len):
        # 此时 past_4_flows_scaled[i] 是一个标量，与 site_encoded 拼接
        step_features = np.concatenate([site_encoded, [past_4_flows_scaled[i]]])
        X.append(step_features)
    
    X_seq = np.array([X])   # 形状变为 (1, 4, n_site_feat + 1)
    
    model = lstm_model if model_name == 'lstm' else gru_model
    pred = model.predict(X_seq, verbose=0)
    return float(pred[0,0])

def flow_to_travel_time(distance_km, flow_veh_per_15min):
    """
    根据流量（veh/15min）和道路长度（km）计算预计通行时间（分钟）。
    使用作业提供的二次方程：flow_per_hour = -1.4648375 * v^2 + 93.75 * v
    其中 flow_per_hour = flow_veh_per_15min * 4
    解出速度 v (km/h)：
        - 若 flow_per_hour <= 351   -> 速度 = 60 (自由流)
        - 若 351 < flow_per_hour <= 1500 -> 取较大根（欠饱和分支）
        - 若 flow_per_hour > 1500       -> 取较小根（过饱和分支）
    """
    flow_per_hour = flow_veh_per_15min * 4.0
    # 自由流阈值（流量低于此值，速度恒为 60）
    if flow_per_hour <= 351.0:
        speed = 60.0
    else:
        a = -1.4648375
        b = 93.75
        c = -flow_per_hour
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            # 数值上不应出现，若出现则使用容量点速度 32
            speed = 32.0
        else:
            v1 = (-b + math.sqrt(discriminant)) / (2*a)   # 较大根（欠饱和）
            v2 = (-b - math.sqrt(discriminant)) / (2*a)   # 较小根（过饱和）
            if flow_per_hour <= 1500.0:
                speed = max(v1, v2)      # 欠饱和分支，取高速度
            else:
                speed = min(v1, v2)      # 过饱和分支，取低速度
        # 限制速度范围 [0, 60]
        speed = min(max(speed, 0.0), 60.0)
    travel_time_min = (distance_km / speed) * 60.0 + 0.5
    return travel_time_min

def predict_travel_time(edge_features, model_name='lstm'):
    """
    edge_features 必须包含:
        - from_site: int
        - to_site: int
        - distance_km: float
        - past_4_flows: list of 4 floats   # 注意：现在是从 from_site 的历史流量
    返回: float 通行时间（分钟）
    """
    # 预测起始站点的未来流量（而不是下游站点）
    future_flow = predict_flow(edge_features['from_site'],
                               edge_features['past_4_flows'],
                               model_name)
    return flow_to_travel_time(edge_features['distance_km'], future_flow)