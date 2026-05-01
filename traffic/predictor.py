# 以后可以切换模型
MODEL_TYPE = "fake"

def predict_traffic(scats_id, time):
    if MODEL_TYPE == "fake":
        import random
        return random.randint(50, 200)

    elif MODEL_TYPE == "lstm":
        # TODO: 调用真实模型
        pass