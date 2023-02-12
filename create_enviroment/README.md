# GPU環境構築

pytorchをgpu環境で使用するには以下のインストールが必要

- driver
- cuda
- cudnn
- pytorch


cuda, cudnnなど、バージョン依存が大きく、バージョンがあっていないと動いてくれないので
注意

## 想定環境　

マシン    : Standard_NC4as_T4_v3
OS       : Ubuntu 20.04 x64 Gen2
driver   :
cuda     : 11.6
cudnn    :  
pytorch  : 

## 初めに

### 使用しているGPUの確認

```bash
lspci | grep -i nvidia
```

Standard_NC4as_T4_v3での結果
```
0001:00:00.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
```

```bash
dpkg -l | grep nvidia
dpkg -l | grep cuda
```

### 現状入っているCUDA nvidiaドライバの削除

他のバージョンがあったりすると、競合して変なインストールエラーが出る.
```bash 
sudo apt-get --purge remove nvidia-*
sudo apt-get --purge remove cuda-*
```

### インストールするdriverのバージョンを確認

```bash
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers devices
```

結果下記のような出力がされる.  `recommended`となっている`nvidia-driver-525 `をインストール. 
```
vendor   : NVIDIA Corporation
model    : TU104GL [Tesla T4]
driver   : nvidia-driver-470 - distro non-free
driver   : nvidia-driver-515 - distro non-free
driver   : nvidia-driver-525 - distro non-free recommended
driver   : nvidia-driver-515-server - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-510 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```


### driverのインストール

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-525 
```
そこそこ時間かかるので待つ. 

### インストールできたら再起動
```
sudo reboot
```
再ログインして、

```
nvidia-smi
```

下記のような結果が出力
```
Sat Dec 17 03:48:17 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000001:00:00.0 Off |                  Off |
| N/A   48C    P8    14W /  70W |     93MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       930      G   /usr/lib/xorg/Xorg                 82MiB |
|    0   N/A  N/A      1079      G   /usr/bin/gnome-shell                7MiB |
+-----------------------------------------------------------------------------+
```

## driverのinstall 

## cudaのインストール

## cudnnのインストール

## pytorchのインストール



