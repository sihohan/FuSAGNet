# -*- coding: utf-8 -*-
import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytz import timezone
from torch.utils.data import DataLoader, Subset

from datasets.TimeDataset import TimeDataset
from evaluate import (
    get_best_performance_data,
    get_full_err_scores,
)
from models.FuSAGNet import FuSAGNet
from train import train
from test import test
from util.net_struct import get_fc_graph_struc, get_feature_map
from util.preprocess import build_loc_net, construct_data


class Main:
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)
        if dataset in ["swat", "wadi"]:
            train, test = (
                train_orig[2160:],
                test_orig,
            )
        else:
            train, test = train_orig, test_orig

        if "attack" in train.columns:
            train = train.drop(columns=["attack"])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        self.device = torch.device(
            f'cuda:{train_config["gpu_id"]}' if torch.cuda.is_available() else "cpu"
        )
        torch.cuda.set_device(self.device)

        fc_edge_index = build_loc_net(
            fc_struc, list(train.columns), feature_map=feature_map
        )
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(
            test, feature_map, labels=test.attack.tolist()
        )

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
        }

        train_dataset = TimeDataset(
            train_dataset_indata,
            fc_edge_index,
            mode="train",
            task="forecasting",
            config=cfg,
        )
        min_train, max_train = train_dataset.get_train_min_max()
        test_dataset = TimeDataset(
            test_dataset_indata,
            fc_edge_index,
            mode="test",
            task="forecasting",
            min_train=min_train,
            max_train=max_train,
            config=cfg,
        )

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset, train_config["batch"], val_ratio=train_config["val_ratio"]
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config["batch"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        if env_config["dataset"] == "hai":
            process_dict = {"P1": 38, "P2": 22, "P3": 7, "P4": 12}
        elif env_config["dataset"] == "swat":
            process_dict = {"P1": 5, "P2": 11, "P3": 9, "P4": 9, "P5": 13, "P6": 4}
        elif env_config["dataset"] == "wadi":
            process_dict = {"P1": 19, "P2": 90, "P3": 15, "P4": 3}

        edge_index_sets = [fc_edge_index]
        self.model = FuSAGNet(
            edge_index_sets=edge_index_sets,
            node_num=len(feature_map),
            dim=train_config["dim"],
            window_size=train_config["slide_win"],
            out_layer_num=train_config["out_layer_num"],
            out_layer_inter_dim=train_config["out_layer_inter_dim"],
            topk=train_config["topk"],
            process_dict=process_dict,
        ).to(self.device)

    def run(self):
        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            model_save_path = self.get_save_path()[0]
            self.train_log = train(
                self.model,
                save_path=model_save_path,
                config=train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                device=self.device,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config["dataset"],
            )

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(
            best_model, self.test_dataloader, device=self.device, config=train_config
        )
        _, self.val_result = test(
            best_model, self.val_dataloader, device=self.device, config=train_config
        )

        self.get_score(
            self.test_result["forecasting"],
            self.test_result["reconstruction"],
            self.val_result["forecasting"],
            self.val_result["reconstruction"],
            self.train_config,
            model_save_path,
            self.env_config["dataset"],
        )

    def get_loaders(self, train_dataset, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len :]]
        )
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(
            train_subset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True
        )
        val_dataloader = DataLoader(
            val_subset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True
        )

        return train_dataloader, val_dataloader

    def get_score(
        self,
        test_result_f,
        test_result_r,
        val_result_f,
        val_result_r,
        config,
        save_path,
        dataset_name,
    ):
        def whm(x1, x2, w1, w2):
            epsilon = 1e-2
            return (w1 + w2) * (x1 * x2) / (w1 * x2 + w2 * x1 + epsilon)

        _, _, test_labels_f = test_result_f
        test_labels = np.asarray(test_labels_f)[:, 0].tolist()

        alpha = config["alpha"]
        test_scores_f, _ = get_full_err_scores(test_result_f[:2], val_result_f[:2])
        test_scores_r, _ = get_full_err_scores(test_result_r[:2], val_result_r[:2])
        test_scores_whm = []
        for i in range(len(test_scores_f)):
            score = whm(
                x1=test_scores_f[i], x2=test_scores_r[i], w1=alpha, w2=1.0 - alpha
            )
            test_scores_whm.append(score)

        all_scores = [test_scores_f, test_scores_r, test_scores_whm]
        score_labels = ["Forecasting", "Reconstruction", "Weighted Harmonic Mean"]
        to_save = {}
        if self.env_config["report"] == "best":
            for i, scores in enumerate(all_scores):
                score_label = score_labels[i]
                if score_label != "Weighted Harmonic Mean":
                    continue

                scores = np.array(scores)
                to_save[score_label] = scores
                top1_best_info = get_best_performance_data(scores, test_labels, topk=1)
                print(
                    f"F1: {top1_best_info[0]:.4f} | Pr: {top1_best_info[1]:.4f} | Re: {top1_best_info[2]:.4f}"
                )

        model_save_name = save_path.split("/")[-1].split(".")[0]
        results_save_path = f"./results/{dataset_name}/{model_save_name}/"
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path, exist_ok=True)

        for k in to_save:
            f2save = to_save.get(k)
            np.save(os.path.join(results_save_path, k), f2save)

    def get_save_path(self, feature_name=""):
        dir_path = self.env_config["dataset"]
        if self.datestr is None:
            now = datetime.now(timezone("Asia/Seoul"))
            self.datestr = now.strftime("%m|%d-%H:%M:%S")

        datestr = self.datestr
        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.csv",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch", help="batch size", type=int, default=32)
    parser.add_argument("-epoch", help="train epoch", type=int, default=50)
    parser.add_argument("-slide_win", help="window size", type=int, default=5)
    parser.add_argument("-dim", help="dimension", type=int, default=64)
    parser.add_argument("-slide_stride", help="window stride", type=int, default=1)
    parser.add_argument("-save_path_pattern", help="save path", type=str, default="")
    parser.add_argument("-dataset", help="hai/swat/wadi", type=str, default="swat")
    parser.add_argument("-device", help="cpu/cuda", type=str, default="cuda")
    parser.add_argument("-random_seed", help="random seed", type=int, default=-999)
    parser.add_argument("-comment", help="experiment comment", type=str, default="")
    parser.add_argument(
        "-out_layer_num", help="out layer dimension", type=int, default=1
    )
    parser.add_argument(
        "-out_layer_inter_dim",
        help="intermediate out layer dimension",
        type=int,
        default=64,
    )
    parser.add_argument("-decay", help="weight decay", type=float, default=0)
    parser.add_argument(
        "-val_ratio", help="validation data ratio", type=float, default=0.2
    )
    parser.add_argument("-topk", help="k", type=int, default=15)
    parser.add_argument("-report", help="best/val", type=str, default="best")
    parser.add_argument(
        "-load_model_path", help="trained model path", type=str, default=""
    )
    parser.add_argument("-lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-gpu_id", help="gpu device ID", type=int, default=1)
    parser.add_argument(
        "-alpha", help="forecasting loss weight", type=float, default=0.5
    )
    parser.add_argument("-beta", help="sparse loss weight", type=float, default=1.0)

    args = parser.parse_args()

    if args.random_seed < 0:
        args.random_seed = random.randint(0, 100)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    train_config = {
        "batch": args.batch,
        "epoch": args.epoch,
        "slide_win": args.slide_win,
        "dim": args.dim,
        "slide_stride": args.slide_stride,
        "comment": args.comment,
        "seed": args.random_seed,
        "out_layer_num": args.out_layer_num,
        "out_layer_inter_dim": args.out_layer_inter_dim,
        "decay": args.decay,
        "val_ratio": args.val_ratio,
        "topk": args.topk,
        "lr": args.lr,
        "gpu_id": args.gpu_id,
        "alpha": args.alpha,
        "beta": args.beta,
    }

    env_config = {
        "save_path": args.save_path_pattern,
        "dataset": args.dataset,
        "report": args.report,
        "device": args.device,
        "load_model_path": args.load_model_path,
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
