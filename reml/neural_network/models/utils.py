import numpy as np


def iterate_minibatches(X, Y, batch_size, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, N, batch_size):
        e = s + batch_size
        b = idx[s:e]
        yield X[b], Y[b]


def fit(model, loss_fn, optimizer, X, Y, epochs=100, batch_size=32, verbose=True):
    for ep in range(1, epochs + 1):
        total = 0.0
        count = 0
        for xb, yb in iterate_minibatches(X, Y, batch_size):
            preds = model.forward(xb)
            loss = loss_fn.forward(preds, yb)
            total += loss * xb.shape[0]
            count += xb.shape[0]
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step(model.params_and_grads())
        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"epoch {ep:4d} | loss {total / count:.6f}")
    return model
