import pandas as pd
import numpy as np
from pathlib import Path
from traffic_predict import predict_flow, predict_travel_time, flow_to_travel_time, predict_travel_time_by_departure

# 1. 加载测试数据（使用你已有的 test.csv，确保包含 flow_target 和滞后特征）
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
df = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['timestamp'])

# 2. 测试 predict_flow 函数
print("=" * 50)
print("测试 predict_flow 函数")
print("=" * 50)

# 随机选取 5 行作为测试样本
samples = df.sample(n=5, random_state=42)

for idx, row in samples.iterrows():
    site = row['site_number']
    past_4 = [row['flow_t_minus_4'], row['flow_t_minus_3'], 
              row['flow_t_minus_2'], row['flow_t_minus_1']]
    true_flow = row['flow_target']
    
    pred_lstm = predict_flow(site, past_4, model_name='lstm')
    pred_gru = predict_flow(site, past_4, model_name='gru')
    
    print(f"站点 {site} 时间 {row['timestamp']}")
    print(f"  过去4个流量: {past_4}")
    print(f"  真实流量: {true_flow:.2f}")
    print(f"  LSTM预测: {pred_lstm:.2f}, 误差: {abs(pred_lstm - true_flow):.2f}")
    print(f"  GRU预测:  {pred_gru:.2f}, 误差: {abs(pred_gru - true_flow):.2f}")
    print("-" * 40)

# 3. 测试 flow_to_travel_time 函数
print("\n" + "=" * 50)
print("测试 flow_to_travel_time 函数")
print("=" * 50)

test_distances = [0.3, 0.5, 1.0]
test_flows_15min = [100, 300, 500, 800, 1200]  # 分别是 400, 1200, 2000, 3200, 4800 veh/h

for dist in test_distances:
    for flow in test_flows_15min:
        tt = flow_to_travel_time(dist, flow)
        print(f"距离 {dist} km, 流量 {flow} veh/15min -> 预计时间 {tt:.2f} 分钟")
    print("-" * 40)

# 4. 测试 predict_travel_time 接口（需要构造 edge_features）
print("\n" + "=" * 50)
print("测试 predict_travel_time 接口")
print("=" * 50)

# 使用前几个测试样本构造 edge_features
for idx, row in samples.iterrows():
    edge = {
        'from_site': row['site_number'],
        'to_site': row['site_number'] + 1,  # 随便填一个终点
        'distance_km': 0.6,
        'past_4_flows': [row['flow_t_minus_4'], row['flow_t_minus_3'], 
                         row['flow_t_minus_2'], row['flow_t_minus_1']]
    }
    tt_lstm = predict_travel_time(edge, model_name='lstm')
    tt_gru = predict_travel_time(edge, model_name='gru')
    print(f"边: {edge['from_site']} -> {edge['to_site']}, 距离 {edge['distance_km']} km")
    print(f"  起始站点过去4流量: {edge['past_4_flows']}")
    print(f"  LSTM预估时间: {tt_lstm:.2f} 分钟")
    print(f"  GRU预估时间:  {tt_gru:.2f} 分钟")
    print("-" * 40)

# 5. 测试 predict_travel_time_by_departure 接口（供 routing 层使用的简化接口）
print("\n" + "=" * 50)
print("测试 predict_travel_time_by_departure 接口（routing 层用）")
print("=" * 50)

for idx, row in samples.iterrows():
    edge = {
        'from_site': row['site_number'],
        'to_site': row['site_number'] + 1,
        'distance_km': 0.6,
        'departure_time': row['timestamp']  # routing 层只需传这四个参数
    }
    tt_lstm = predict_travel_time_by_departure(edge, model_name='lstm')
    tt_gru = predict_travel_time_by_departure(edge, model_name='gru')
    print(f"边: {edge['from_site']} -> {edge['to_site']}, 距离 {edge['distance_km']} km")
    print(f"  出发时间: {edge['departure_time']}")
    print(f"  LSTM预估时间: {tt_lstm:.2f} 分钟")
    print(f"  GRU预估时间:  {tt_gru:.2f} 分钟")
    print("-" * 40)

print("\n测试完成。")