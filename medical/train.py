from pathlib import Path
from data import create_valid_files, C20Dataset, C20DatasetOnMemory
import pandas as pd
from conv import CondNet1D
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

import torch
import torch.optim as optim
from torch import nn
import time

class Trainer:
    def __init__(self, model, device) -> None:
        
        self.model = model.to(device)
        self.device = device
        logging.info(f"device is {self.device}")

        self.lossfunc = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_one_epoch(self, trainloader):
        self.model.train()
        running_loss = 0.0
        for X, y in trainloader:
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()

            pred= self.model(X)
            loss = self.lossfunc(pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(trainloader)

    def valid(self, testloader):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for X, y in testloader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred= self.model(X)
                loss = self.lossfunc(pred, y)
                valid_loss += loss.item()
        return valid_loss / len(testloader)

    def train(self, trainloader, testloader, epochs):

        logging.info(" ----- train start ------ ")
        logging.info(f"epoch size = {epochs}")

        for ep in range(epochs):
            st = time.time()
            train_loss = self.train_one_epoch(trainloader)
            et = time.time() - st
            valid_loss = self.valid(testloader)
            logging.info(
                f"epoch {ep} : train loss = {train_loss:.4f} "
                f"valid loss = {valid_loss:.4f} "
                f"train time : {et:.2f} [s]"
                )

if __name__ == "__main__":
    logging.basicConfig(filename="log/sample.log",encoding='utf-8', level=logging.INFO)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    basepath = Path("/mnt/data/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/")
    headers = list(basepath.glob("**/*.hea"))

    logging.info(f"EDG file num: {len(headers)}")
    df = create_valid_files(headers, ncpu=16)
    train_rate = 0.8
    train_size = int(df.shape[0] * train_rate)
    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]

    logging.info(f"train size : {train_df.shape[0]}")
    logging.info(f"test size : {test_df.shape[0]}")

    train_df.to_csv("data/train.csv", index=0)
    test_df.to_csv("data/test.csv", index=0)

    trainset = C20DatasetOnMemory("data/train.csv", 12, 1000, 27)
    testset = C20DatasetOnMemory("data/test.csv", 12, 1000, 27)

    trainloader = DataLoader(trainset, batch_size=512, shuffle=True)
    testloader = DataLoader(testset, batch_size=512, shuffle=True)

    model = CondNet1D(12, 27)

    trainer = Trainer(model, device)
    trainer.train(trainloader, testloader, 500)
