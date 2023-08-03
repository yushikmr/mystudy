echo "############# APT UPDATE & UPGRADE #####################"
sudo apt update -y
sudo apt upgrade -y

echo "INSTALL GIT"

sudo apt install -y build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev tk-dev git

git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv
git checkout v2.0.3

# .bashrcの更新
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
source ~/.bashrc
