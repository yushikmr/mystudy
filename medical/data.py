import wfdb
from pathlib import Path
import re
import json
import numpy as np
from typing import List, Tuple
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from multiprocessing import Pool
import logging


class C20Patient:


    def __init__(self, record_path:Path) -> None:
        
        self.get_cfg()
        self.path = record_path
        self.record_name:str  = os.path.splitext(str(record_path))[0]
        self.record = wfdb.rdrecord(self.record_name)
        self.header = wfdb.rdheader(self.record_name)
        self.fs:int = self.header.fs
        self.source:str = record_path.parent.parent.name
        self.samplingrate = int(self.fs / self.freq)
        self._df = self.record.to_dataframe()\
                .iloc[::self.samplingrate]
        self._df = self.df.iloc[:self.samplingsize].fillna(0)
        self._get_Xnp()
        
    
    def _get_labelids(self, record_path):
        with open(record_path, "r") as f:
            text = f.readlines()
        line =  [t for t in text if "Dx" in t][0]
        _labelids:List[str] = re.findall(r'\d+', line)
        self._labelids:List[str] = [lid for lid in _labelids 
                                        if lid in self._labelsdict.keys()]

    @property
    def df(self)->pd.DataFrame:
        return self._df

    def get_labelID(self):
        if not hasattr(self, "_labelid"):
            self._get_labelids(self.path)
        return self._labelids
    
    @property
    def get_label(self)->List[str]:
        if not hasattr(self, "_labelid"):
            self._get_labelids(self.path)
        return [self._labelsdict[i] for i in self._labelids]

    @property
    def Xnp(self):
        return self._Xnp

    @property
    def Ynp(self):
        ynp = np.zeros(len(self.labelnames), dtype=np.float32)
        for l in self.get_label:
            ind = self.labelnames.index(l)
            ynp[ind] = 1
        return ynp

    def _get_Xnp(self):
        self._Xnp =  self.df.values.astype(np.float32)

    @property
    def toTensorX(self)->torch.Tensor:
        return torch.from_numpy(self._Xnp).permute(1, 0)

    @property
    def toTensorY(self)->torch.Tensor:
        return torch.from_numpy(self.Ynp)

    def checklabel(self)->bool:
        hasvalidlabel = self.Ynp.sum() > 0.0
        hasenoughtsize = self.df.shape[0] >= self.samplingsize
        if hasvalidlabel and hasenoughtsize:
            return True
        else:
            return False
    

    def get_cfg(self):
        cfgpath = Path(__file__).parent / "data.json"
        with open(cfgpath, "r") as f:
            cfg = json.load(f)["challenge-2020"]
        self.freq = cfg["freq"]
        self.ecg_type = cfg["ecg_type"] 
        self.samplingsize = cfg["samplingsize"]
        self._labelsdict:dict = cfg["labels"]
        self.labelnames:list = cfg["labelnames"]
        

class C20:
    def __init__(self, patients:List[C20Patient], valid=True) -> None:
        
        self.get_cfg()
        if valid:
            self._patients = self._get_valid_patients(patients)
        else:
            self._patients = patients
        self._get_Xnp()
        self._get_labels()
    
    def _get_valid_patients(self, patients:List[C20Patient])->List[C20Patient]:
        valid_patients = \
        [patient for patient in patients 
                 if patient.get_labelID() in self._valid_ids]
        if len(valid_patients) == 0:
            raise ValueError("there are no valid patient.")
        return valid_patients


    def _get_labels(self):
        label_list = [self._labeldict[p.get_labelID()] for p in self._patients]
        self._labels = pd.get_dummies(pd.DataFrame(label_list))

    def _get_Xnp(self):
        x_list = []
        for patient in self._patients:
            x_list.append(patient.df.values[np.newaxis, :, :])
        self._Xnp =  np.concatenate(x_list)


    @property
    def label(self):
        return self._labels

    def get_uniq_labels(self):
        return set([self._labeldict[p.get_labelID()]for p in self._patients])

    @property
    def Xnp(self)->np.ndarray:
        return self._Xnp.astype(np.float32)

    @property
    def Ynp(self)->np.ndarray:
        return self._labels.values.astype(np.float32)

    def toTensorX(self)->torch.Tensor:
        return torch.from_numpy(self.Xnp).permute(0, 2, 1)

    def toTensorY(self)->torch.Tensor:
        return torch.from_numpy(self.Ynp)

    def get_cfg(self):
        cfgpath = Path(__file__).parent / "data.json"
        with open(cfgpath, "r") as f:
            cfg = json.load(f)["challenge-2020"]
        self.freq = cfg["freq"]
        self.ecg_type = cfg["ecg_type"] 
        self.samplingsize = cfg["samplingsize"]
        self._valid_ids = list(cfg["labels"].keys())
        self._labeldict = cfg["labels"]


def get_annotaion(header):
    c = C20Patient(header)
    if c.checklabel():
        return [str(header), c.get_label]

def create_valid_files(header_list:List[Path], ncpu:int=1)->pd.DataFrame:
    
    with Pool(ncpu) as p:
        imap = p.map(get_annotaion, header_list)
    annotaions = list(imap)
    annotaions = [anno for anno in annotaions if anno is not None]
    return pd.DataFrame(annotaions, columns=["file", "label"])

class C20Dataset(Dataset):
    def __init__(self, valid_files_csv) -> None:
        self.patients_files = pd.read_csv(valid_files_csv)
        self._len = self.patients_files.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index:int) -> Tuple[torch.Tensor]:
    
        file = Path(self.patients_files.iloc[index, 0])
        patient = C20Patient(file)
        X = patient.toTensorX
        y = patient.toTensorY
        return X, y
        
class C20DatasetOnMemory(Dataset):

    def __init__(
            self, 
            valid_files_csv, 
            num_channels, 
            sample_size, 
            num_classes) -> None:
        """"""
        logging.info("Dataset on Memory.........")
        self.patients_files = pd.read_csv(valid_files_csv)
        logging.info(f"num of patients : {self.patients_files.shape[0]}")
        self._len = self.patients_files.shape[0]

        self.num_channels = num_channels
        self.sample_size = sample_size
        self.num_classes = num_classes

        self._create_tensor()

    def __len__(self):
        return self._len
     
    def _create_tensor(self):
        logging.info("loading all data .............")
        X_list = []
        y_list = []
        for i in range(self.patients_files.shape[0]):
            f = Path(self.patients_files.iloc[i, 0])
            p = C20Patient(f)
            X_list.append(p.toTensorX.reshape(1, self.num_channels, self.sample_size))
            y_list.append(p.toTensorY.reshape(1, self.num_classes))
            
        self.X = torch.cat(X_list, dim=0)
        self.y = torch.cat(y_list, dim=0) 
        logging.info("done loading all data .............") 

    def __getitem__(self, index) -> Tuple[torch.Tensor]:

        return self.X[index], self.y[index]

     
    
