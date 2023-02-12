# /bin/bash

wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"


[ -f  "${HOME}/conda/etc/profile.d/conda.sh" ] && source "${HOME}/conda/etc/profile.d/conda.sh"
[ -f  "${HOME}/conda/etc/profile.d/mamba.sh" ]  && source "${HOME}/conda/etc/profile.d/mamba.sh"

