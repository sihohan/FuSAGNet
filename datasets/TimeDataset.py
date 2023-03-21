import torch
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    def __init__(
        self,
        raw_data,
        edge_index,
        mode="train",
        task="reconstruction",
        min_train=None,
        max_train=None,
        config=None,
        preprocess=None,
    ):
        self.raw_data = raw_data
        self.edge_index = edge_index
        self.mode = mode
        self.task = task
        self.config = config

        x_data = raw_data[:-1]
        labels = raw_data[-1]
        data = torch.tensor(x_data).double()
        if mode == "train":
            min_train, max_train = [], []
            for i in range(data.size(0)):
                col_min, col_max = torch.min(data[i]), torch.max(data[i])
                min_train.append(col_min)
                max_train.append(col_max)
                if (col_min == 0.0) and (col_max == 0.0):
                    pass
                elif (col_min != 0.0) and (col_max == col_min):
                    data[i] /= min_train[-1]
                else:
                    data[i] -= min_train[-1]
                    data[i] /= max_train[-1] - min_train[-1]

            self.min_train, self.max_train = min_train, max_train

        elif mode == "test":
            for i in range(data.size(0)):
                col_min, col_max = min_train[i], max_train[i]
                if (col_min == 0.0) and (col_max == 0.0):
                    pass
                elif (col_min != 0.0) and (col_max == col_min):
                    data[i] /= col_min
                else:
                    data[i] -= col_min
                    data[i] /= col_max - col_min

        labels = torch.tensor(labels).double()
        self.x, self.y, self.labels = self.process(data, labels)

    def get_train_min_max(self):
        return self.min_train, self.max_train

    def get_train_mean_std(self):
        return self.means, self.stds

    def process(self, data, labels):
        x_arr, labels_arr = [], []
        y_arr = None if self.task == "reconstruction" else []
        slide_win, slide_stride = [
            self.config[k] for k in ("slide_win", "slide_stride")
        ]
        is_train = self.mode == "train"
        _, total_time_len = data.shape
        rang = (
            range(slide_win, total_time_len, slide_stride)
            if is_train
            else range(slide_win, total_time_len)
        )
        for i in rang:
            window = data[:, i - slide_win : i]
            if y_arr is not None:
                target = data[:, i]
                y_arr.append(target)

            x_arr.append(window)
            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous() if y_arr is not None else None
        labels = torch.Tensor(labels_arr).contiguous()
        return x, y, labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        window = self.x[idx].double()
        window_y = self.y[idx].double() if self.y is not None else None
        label = self.labels[idx].double()
        edge_index = self.edge_index.long()
        return window, window_y, label, edge_index
