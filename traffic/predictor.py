MODEL_TYPE = "fake"


def predict_traffic(scats_id, time):
    if MODEL_TYPE == "fake":
        import random
        return random.randint(50, 200)

    elif MODEL_TYPE == "lstm":
        return 100