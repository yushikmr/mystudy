from pathlib import Path
from data import C20, C20Patient, create_valid_files, C20Dataset
import pandas as pd
from conv import CondNet1D
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

import torch
import torch.optim as optim
from torch import nn

class Trainer:
    def __init__(self, model) -> None:
        
        self.model = model

        self.lossfunc = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_one_epoch(self, trainloader):
        self.model.train()
        running_loss = 0.0
        for X, y in tqdm(trainloader):
            
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
            for X, y in tqdm(testloader):
                self.optimizer.zero_grad()
                pred= self.model(X)
                loss = self.lossfunc(pred, y)
                valid_loss += loss.item()
        return valid_loss / len(testloader)

    def train(self, trainloader, testloader, epochs):

        logging.info(" ----- train start ------ ")
        logging.info(f"epoch size = {epochs}")

        for ep in range(epochs):
            train_loss = self.train_one_epoch(trainloader)
            valid_loss = self.valid(testloader)
            logging.info(
                f"epoch {ep} : train loss = {train_loss} "
                f"valid loss = {valid_loss}"
                )

if __name__ == "__main__":
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    basepath = Path("/data/dummy/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/")
    headers = list(basepath.glob("**/*.hea"))[:100]
    df = create_valid_files(headers, ncpu=16)
    train_rate = 0.8
    train_size = int(df.shape[0] * train_rate)
    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]

    train_df.to_csv("data/train.csv", index=0)
    test_df.to_csv("data/test.csv", index=0)

    trainset = C20Dataset("data/train.csv")
    testset = C20Dataset("data/test.csv")

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    model = CondNet1D(12, 27)

    trainer = Trainer(model)
    trainer.train(trainloader, testloader, 10)
