import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ==========================================
# 1. 路径设置 (使用绝对路径或基于项目根目录的路径)
# ==========================================
# 获取当前脚本所在目录的父目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'test.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'onehot_encoder.pkl')
LSTM_PATH = os.path.join(MODEL_DIR, 'lstm_best.keras')
GRU_PATH = os.path.join(MODEL_DIR, 'gru_best.keras')
RF_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

SEQ_LEN = 4  # 必须与训练 LSTM/GRU 时一致

# ==========================================
# 2. 数据准备函数
# ==========================================
def load_and_prepare_test_data(file_path, scaler, enc):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到测试集文件: {file_path}")
        
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    site_col = 'site_number'
    target_col = 'flow_target'
    
    X_windows = []
    y_true = []
    site_ids = []
    
    # 按站点分组构造序列，确保与训练时逻辑一致
    for site, group in df.groupby(site_col):
        group = group.sort_values('timestamp')
        flows = group[target_col].values
        if len(flows) <= SEQ_LEN:
            continue
        for i in range(len(flows) - SEQ_LEN):
            X_windows.append(flows[i:i+SEQ_LEN])
            y_true.append(flows[i+SEQ_LEN])
            site_ids.append(site)
    
    X_seq = np.array(X_windows)
    y_true = np.array(y_true)
    site_ids = np.array(site_ids).reshape(-1, 1)
    
    # 站点 One-Hot 编码
    site_encoded = enc.transform(site_ids)
    n_site_feat = site_encoded.shape[1]
    
    # 构造 LSTM/GRU 输入 (samples, SEQ_LEN, n_site_feat + 1)
    X_rnn = []
    for i in range(len(X_seq)):
        sample = []
        for t in range(SEQ_LEN):
            step = np.concatenate([site_encoded[i], [X_seq[i, t]]])
            sample.append(step)
        X_rnn.append(sample)
    X_rnn = np.array(X_rnn)
    
    # 标准化流量数值部分
    numeric_part = X_rnn[:, :, n_site_feat:]
    shape = numeric_part.shape
    X_rnn[:, :, n_site_feat:] = scaler.transform(numeric_part.reshape(-1, 1)).reshape(shape)
    
    # 构造随机森林输入 (通常 RF 使用平铺的特征)
    # 假设 RF 输入是: [site_one_hot..., flow_t-4, flow_t-3, flow_t-2, flow_t-1]
    # 注意：RF 通常不需要标准化，但如果训练时用了，这里也要用。这里我们保持原始流量值。
    X_rf = []
    for i in range(len(X_seq)):
        # 拼接站点编码和过去4个时间步的原始流量
        feat = np.concatenate([site_encoded[i], X_seq[i]])
        X_rf.append(feat)
    X_rf = np.array(X_rf)
    
    return X_rnn, X_rf, y_true

def main():
    print("正在加载模型和数据...")
    
    # 检查文件是否存在
    required_files = [SCALER_PATH, ENCODER_PATH, LSTM_PATH, GRU_PATH]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 缺失必要文件 {f}")
            return

    # 加载辅助工具
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    # 加载数据
    X_rnn, X_rf, y_true = load_and_prepare_test_data(DATA_PATH, scaler, encoder)
    print(f"测试样本数量: {len(y_true)}")

    # 1. LSTM 预测
    print("执行 LSTM 预测...")
    lstm_model = load_model(LSTM_PATH)
    y_pred_lstm = lstm_model.predict(X_rnn, verbose=0).flatten()
    
    # 2. GRU 预测
    print("执行 GRU 预测...")
    gru_model = load_model(GRU_PATH)
    y_pred_gru = gru_model.predict(X_rnn, verbose=0).flatten()
    
    # 3. Random Forest 预测
    y_pred_rf = None
    if os.path.exists(RF_PATH):
        print("执行 Random Forest 预测...")
        rf_model = joblib.load(RF_PATH)
        y_pred_rf = rf_model.predict(X_rf)
    else:
        print("\n" + "!"*50)
        print("警告: 找不到随机森林模型文件 rf_model.pkl")
        print(f"请将同学那里的模型文件放入路径: {RF_PATH}")
        print("目前将跳过随机森林，仅计算 LSTM 和 GRU 的集成。")
        print("!"*50 + "\n")

    # ==========================================
    # 3. 计算集成预测 (算术平均)
    # ==========================================
    if y_pred_rf is not None:
        y_pred_ensemble = (y_pred_lstm + y_pred_gru + y_pred_rf) / 3.0
        models_to_compare = {
            "LSTM": y_pred_lstm,
            "GRU": y_pred_gru,
            "Random Forest": y_pred_rf,
            "Simple Ensemble": y_pred_ensemble
        }
    else:
        # 如果没有 RF，只做 LSTM + GRU 的集成
        y_pred_ensemble = (y_pred_lstm + y_pred_gru) / 2.0
        models_to_compare = {
            "LSTM": y_pred_lstm,
            "GRU": y_pred_gru,
            "Simple Ensemble (LSTM+GRU)": y_pred_ensemble
        }

    # ==========================================
    # 4. 计算 MAE 并比较
    # ==========================================
    print("\n" + "="*40)
    print(f"{'Model':<25} | {'MAE':<10}")
    print("-" * 40)
    
    results = []
    for name, y_pred in models_to_compare.items():
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{name:<25} | {mae:<10.4f}")
        results.append((name, mae))
    print("="*40)

    # ==========================================
    # 5. 可视化结果与保存
    # ==========================================
    names = [r[0] for r in results]
    maes = [r[1] for r in results]
    
    # 保存结果到 CSV
    res_df = pd.DataFrame({"Model": names, "MAE": maes})
    csv_save_path = os.path.join(BASE_DIR, 'data', 'model_output', 'ensemble_results.csv')
    res_df.to_csv(csv_save_path, index=False)
    print(f"结果已保存至: {csv_save_path}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, maes, color=['skyblue', 'lightgreen', 'orange', 'salmon'])
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Comparison: Individual vs Ensemble')
    
    # 在柱状图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 4), ha='center', va='bottom')
    
    save_fig_path = os.path.join(BASE_DIR, 'docs', 'figures', 'ensemble_comparison.png')
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path)
    print(f"\n对比图表已保存至: {save_fig_path}")
    plt.show()

if __name__ == "__main__":
    main()
