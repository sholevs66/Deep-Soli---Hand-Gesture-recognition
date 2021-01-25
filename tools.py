def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def shuffle_data(x, y, y_l, batch_size):
    import torch
    idx_train_shuffle = torch.randperm(x.shape[0])
    x = x[idx_train_shuffle, :]
    x_ch = torch.split(x, batch_size, dim=0)
    x_ch = x_ch[0:-1]
    idx=list(idx_train_shuffle.numpy())
    y = y[idx_train_shuffle, :]
    y_l = y_l[idx_train_shuffle]
    y_ch = torch.split(y, batch_size, dim=0)
    y_ch = y_ch[0:-1]

    return x, y, x_ch, y_ch, y_l

def calc_error(y_pred, y_true):
    import torch
    return (torch.count_nonzero(y_pred - y_true) / y_pred.shape[0]).cpu().data.numpy()


def get_last_element(y, N):
    return [y[i][-1] for i in [*range(0, N, 1)]]