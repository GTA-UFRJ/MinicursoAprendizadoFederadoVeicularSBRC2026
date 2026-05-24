import torch
from collections import deque
from .train import LSTM, load_tp, create_dataset

model = LSTM()

model.load_state_dict(torch.load("models/model_10.pt", weights_only=True))

tpu, tpd = load_tp()

# train-test split for time series
train_size = int(len(tpd) * 0.67)
test_size = len(tpd) - train_size
train, test = tpd[:train_size], tpd[train_size:]

lookback = 10
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

s  = deque(10*[100],10)
l = list(s)
window = torch.tensor(l,dtype=torch.float32).view(-1,1)
print("x[0]: ",X_test[0])
print("x[0]: ",X_test[0].shape)
print(window.shape)
print("window: ", window)

#print("y_hat: ",float(model(X_test[0])[-1]))
print("y_hat: ", model(window))
print(" y: ",y_test[0])
