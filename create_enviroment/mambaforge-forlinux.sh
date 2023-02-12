# /bin/bash

wget -O Mambaforge3.sh  "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge3.sh -b -p "${HOME}/conda"


[ -f  "${HOME}/conda/etc/profile.d/conda.sh" ] && source "${HOME}/conda/etc/profile.d/conda.sh"
[ -f  "${HOME}/conda/etc/profile.d/mamba.sh" ]  && source "${HOME}/conda/etc/profile.d/mamba.sh"

