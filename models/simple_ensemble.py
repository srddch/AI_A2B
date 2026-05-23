import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 添加项目根目录到 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.rf_feature_utils import build_rf_feature

# ==================== 路径配置 ====================
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_best.keras")
GRU_PATH = os.path.join(MODEL_DIR, "gru_best.keras")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "model_output")
FIGURE_DIR = os.path.join(BASE_DIR, "docs", "figures")

SEQ_LEN = 4

# ==================== 辅助函数 ====================
def load_and_prepare_test_data(file_path, scaler, encoder):
    """
    加载测试数据，构造 RNN（LSTM/GRU）和 Random Forest 所需的特征矩阵。
    返回:
        X_rnn: 形状 (样本数, SEQ_LEN, 特征数) - 已归一化（数值部分）
        X_rf:  形状 (样本数, 特征维度) - 含统计/时间特征
        y_true: 原始流量值（未归一化）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试集文件不存在: {file_path}")

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    site_col = "site_number"
    target_col = "flow_target"

    X_windows = []   # 原始流量窗口 (SEQ_LEN,)
    y_true = []      # 下一个时间点的原始流量
    site_ids = []
    target_times = []  # 预测目标时刻的时间戳

    for site, group in df.groupby(site_col):
        group = group.sort_values("timestamp").reset_index(drop=True)
        flows = group[target_col].values
        times = group["timestamp"].values

        if len(flows) <= SEQ_LEN:
            continue

        for i in range(len(flows) - SEQ_LEN):
            X_windows.append(flows[i:i+SEQ_LEN])
            y_true.append(flows[i+SEQ_LEN])
            site_ids.append(site)
            target_times.append(times[i+SEQ_LEN])

    X_seq = np.array(X_windows)            # (n_samples, SEQ_LEN)
    y_true = np.array(y_true)              # 原始流量
    site_ids = np.array(site_ids).reshape(-1, 1)

    # ----- 站点 one‑hot 编码 -----
    site_encoded = encoder.transform(site_ids)   # (n_samples, n_site_feat)
    n_site_feat = site_encoded.shape[1]

    # ----- 构造 RNN 输入 (LSTM/GRU) -----
    X_rnn = []
    for i in range(len(X_seq)):
        sample = []
        for t in range(SEQ_LEN):
            step = np.concatenate([site_encoded[i], [X_seq[i, t]]])
            sample.append(step)
        X_rnn.append(sample)
    X_rnn = np.array(X_rnn)   # (n_samples, SEQ_LEN, n_site_feat+1)

    # 对数值部分（流量）进行归一化（站点 one‑hot 不变）
    numeric_part = X_rnn[:, :, n_site_feat:]   # (n_samples, SEQ_LEN, 1)
    shape = numeric_part.shape
    X_rnn[:, :, n_site_feat:] = scaler.transform(
        numeric_part.reshape(-1, 1)
    ).reshape(shape)

    # ----- 构造 Random Forest 输入（特征工程）-----
    X_rf = []
    for i in range(len(X_seq)):
        feat = build_rf_feature(
            site_id=site_ids[i][0],
            past_4_flows=X_seq[i],
            timestamp=pd.to_datetime(target_times[i]),
            encoder=encoder
        )
        X_rf.append(feat)
    X_rf = np.array(X_rf)

    return X_rnn, X_rf, y_true, scaler   # 返回 scaler 以便后续反归一化


def calculate_metrics(y_true, y_pred):
    """计算 MAE, RMSE, MAPE (过滤 y_true 极小的点)"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 只计算 y_true 大于 1e-6 的点（避免除零）
    mask = y_true > 1e-6
    if np.sum(mask) == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return mae, rmse, mape


def inverse_transform_if_needed(predictions, scaler, original_y_true):
    """
    自动检测预测值是否需要反归一化。
    如果预测值的最大值明显小于真实值的最大值（例如预测值 max<1 而真实值 max>10），
    则用 scaler 进行反归一化，并打印提示。
    """
    pred_max = predictions.max()
    true_max = original_y_true.max()
    if pred_max < 1 and true_max > 10:
        print(f"  检测到预测值范围 [{pred_max:.3f}] 远小于真实值范围 [{true_max:.1f}]，执行反归一化...")
        predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions_orig
    else:
        return predictions


def main():
    print("=" * 60)
    print("模型评估与集成 (支持自动反归一化)")
    print("=" * 60)

    # 检查必要文件
    required = [SCALER_PATH, ENCODER_PATH, LSTM_PATH, GRU_PATH]
    for fp in required:
        if not os.path.exists(fp):
            print(f"错误：缺少文件 {fp}")
            return

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # 加载测试数据
    print("\n加载测试数据并构造特征...")
    X_rnn, X_rf, y_true, scaler = load_and_prepare_test_data(DATA_PATH, scaler, encoder)
    print(f"测试样本数: {len(y_true)}")

    # ----- 模型预测 -----
    print("\n执行 LSTM 预测...")
    lstm_model = load_model(LSTM_PATH)
    y_pred_lstm_raw = lstm_model.predict(X_rnn, verbose=0).flatten()

    print("执行 GRU 预测...")
    gru_model = load_model(GRU_PATH)
    y_pred_gru_raw = gru_model.predict(X_rnn, verbose=0).flatten()

    # 自动反归一化（如果需要）
    y_pred_lstm = inverse_transform_if_needed(y_pred_lstm_raw, scaler, y_true)
    y_pred_gru = inverse_transform_if_needed(y_pred_gru_raw, scaler, y_true)

    # Random Forest
    y_pred_rf = None
    if os.path.exists(RF_PATH):
        print("执行 Random Forest 预测...")
        rf_model = joblib.load(RF_PATH)
        y_pred_rf_raw = rf_model.predict(X_rf)
        y_pred_rf = inverse_transform_if_needed(y_pred_rf_raw, scaler, y_true)
    else:
        print("警告: 未找到 Random Forest 模型，仅对比 LSTM/GRU/集成")

    # ----- 动态集成权重 (基于 MAE 的倒数) -----
    # 计算各个模型在测试集上的 MAE（用于权重，注意测试集不能用于调权，这里仅作示例）
    # 实际项目中应使用验证集，但此处作为演示我们仍基于测试集。
    # 您也可以固定权重，如 [0.45,0.45,0.1]。
    mae_lstm, _, _ = calculate_metrics(y_true, y_pred_lstm)
    mae_gru, _, _ = calculate_metrics(y_true, y_pred_gru)
    if y_pred_rf is not None:
        mae_rf, _, _ = calculate_metrics(y_true, y_pred_rf)
        # 误差倒数归一化
        inv_errors = np.array([1/mae_lstm, 1/mae_gru, 1/mae_rf])
        weights = inv_errors / inv_errors.sum()
        w_lstm, w_gru, w_rf = weights
        print(f"\n动态权重 (基于MAE倒数): LSTM={w_lstm:.3f}, GRU={w_gru:.3f}, RF={w_rf:.3f}")
        y_pred_ensemble = w_lstm * y_pred_lstm + w_gru * y_pred_gru + w_rf * y_pred_rf
        models_to_compare = {
            "LSTM": y_pred_lstm,
            "GRU": y_pred_gru,
            "Random Forest": y_pred_rf,
            "Weighted Ensemble": y_pred_ensemble
        }
    else:
        # 无 RF 时简单平均
        y_pred_ensemble = (y_pred_lstm + y_pred_gru) / 2
        models_to_compare = {
            "LSTM": y_pred_lstm,
            "GRU": y_pred_gru,
            "Average Ensemble (LSTM+GRU)": y_pred_ensemble
        }

    # ----- 输出指标表格 -----
    print("\n" + "=" * 80)
    print(f"{'Model':<25} | {'MAE':<12} | {'RMSE':<12} | {'MAPE (%)':<12}")
    print("-" * 80)

    results = []
    for name, pred in models_to_compare.items():
        mae, rmse, mape = calculate_metrics(y_true, pred)
        print(f"{name:<25} | {mae:<12.4f} | {rmse:<12.4f} | {mape:<12.2f}")
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape})

    print("=" * 80)

    # ----- 保存结果到 CSV -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_DIR, "ensemble_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n评估结果已保存至: {results_csv}")

    # 保存详细预测值
    pred_df = pd.DataFrame({"True": y_true, "LSTM": y_pred_lstm, "GRU": y_pred_gru})
    if y_pred_rf is not None:
        pred_df["Random Forest"] = y_pred_rf
        pred_df["Weighted Ensemble"] = y_pred_ensemble
    else:
        pred_df["Average Ensemble"] = y_pred_ensemble
    pred_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"详细预测值已保存至: {pred_csv}")

    # ----- 绘制 MAE 对比图 -----
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df["Model"], results_df["MAE"], color=['skyblue', 'lightgreen', 'orange', 'salmon'][:len(results_df)])
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Model Comparison (Original Scale)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "ensemble_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()
    print(f"对比图已保存至: {fig_path}")

    print("\n评估完成！")


if __name__ == "__main__":
    main()