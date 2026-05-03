import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import joblib

DATA_DIR = "./"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"
SEQ_LEN = 4          # 使用过去4个时间步
BATCH_SIZE = 64
EPOCHS = 50

def load_and_prepare_data(file_path, scaler=None, enc=None, fit_scaler=False):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    # 需要的列：站点、目标流量、历史流量（用于构造滑动窗口）
    # 注意：这里我们使用原始的 flow_target 列来构造序列，而不是滞后列
    site_col = 'site_number'
    target_col = 'flow_target'
    
    # 按站点分组，每个站点按时间排序
    X_windows = []
    y_windows = []
    site_ids = []
    
    for site, group in df.groupby(site_col):
        group = group.sort_values('timestamp')
        flows = group[target_col].values
        if len(flows) <= SEQ_LEN:
            continue
        for i in range(len(flows) - SEQ_LEN):
            X_windows.append(flows[i:i+SEQ_LEN])   # 过去4个时间步的流量
            y_windows.append(flows[i+SEQ_LEN])     # 下一个时间步的流量
            site_ids.append(site)
    
    X_seq = np.array(X_windows)   # (samples, SEQ_LEN)
    y_seq = np.array(y_windows)   # (samples,)
    site_ids = np.array(site_ids).reshape(-1, 1)
    
    # 对站点进行 one-hot 编码
    if fit_scaler:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        site_encoded = enc.fit_transform(site_ids)
    else:
        site_encoded = enc.transform(site_ids)
    
    # 将站点编码与流量序列合并：每个样本的输入形状为 (SEQ_LEN, n_site_feat+1)
    # 方式：对每个样本，每个时间步都拼接相同的站点编码
    n_site_feat = site_encoded.shape[1]
    X_combined = []
    for i in range(len(X_seq)):
        sample = []
        for t in range(SEQ_LEN):
            step = np.concatenate([site_encoded[i], [X_seq[i, t]]])
            sample.append(step)
        X_combined.append(sample)
    X_combined = np.array(X_combined)   # (samples, SEQ_LEN, n_site_feat+1)
    
    # 标准化数值部分（即每个时间步的流量值）
    numeric_part = X_combined[:, :, n_site_feat:]   # shape (samples, SEQ_LEN, 1)
    numeric_flat = numeric_part.reshape(-1, 1)
    if fit_scaler:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_flat)
    else:
        numeric_scaled = scaler.transform(numeric_flat)
    X_combined[:, :, n_site_feat:] = numeric_scaled.reshape(numeric_part.shape)
    
    return X_combined, y_seq, scaler, enc

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    print("加载训练集...")
    X_train, y_train, scaler, enc = load_and_prepare_data(
        os.path.join(DATA_DIR, TRAIN_FILE), fit_scaler=True)
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    
    print("加载验证集...")
    X_val, y_val, _, _ = load_and_prepare_data(
        os.path.join(DATA_DIR, VAL_FILE), scaler=scaler, enc=enc, fit_scaler=False)
    print(f"验证集形状: {X_val.shape}, 标签形状: {y_val.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 训练 LSTM
    print("\n训练 LSTM ...")
    lstm_model = build_lstm_model(input_shape)
    lstm_model.summary()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('lstm_best.keras', save_best_only=True, monitor='val_loss')
    ]
    lstm_model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   callbacks=callbacks,
                   verbose=1)
    lstm_model.save('lstm_model.keras')
    print("LSTM 模型已保存为 lstm_model.keras")
    
    # 训练 GRU
    print("\n训练 GRU ...")
    gru_model = build_gru_model(input_shape)
    gru_model.summary()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('gru_best.keras', save_best_only=True, monitor='val_loss')
    ]
    gru_model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=callbacks,
                  verbose=1)
    gru_model.save('gru_model.keras')
    print("GRU 模型已保存为 gru_model.keras")
    
    # 测试集评估
    X_test, y_test, _, _ = load_and_prepare_data(
        os.path.join(DATA_DIR, TEST_FILE), scaler=scaler, enc=enc, fit_scaler=False)
    lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test, verbose=0)
    gru_loss, gru_mae = gru_model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM 测试集 - Loss: {lstm_loss:.4f}, MAE: {lstm_mae:.4f}")
    print(f"GRU 测试集  - Loss: {gru_loss:.4f}, MAE: {gru_mae:.4f}")
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(enc, 'onehot_encoder.pkl')
    print("scaler 和 encoder 已保存。")

if __name__ == "__main__":
    main()