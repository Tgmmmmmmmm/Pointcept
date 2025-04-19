# 首先需要安装conda、git
# 如果服务器不能连接外网，需要进行git代理设置
# git config --global http.proxy http://192.168.31.2:7890
# git config --global https.proxy https://192.168.31.2:7890

conda env create -f environment.yml -vv
conda activate pointcept

# 如果pip的时候不动，调试安装
pip install --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html   -v torch-cluster torch-scatter torch-sparse torch-geometric   spconv-cu124   git+https://github.com/octree-nn/ocnn-pytorch.git   git+https://github.com/openai/CLIP.git   git+https://github.com/Dao-AILab/flash-attention.git   ./libs/pointops   ./libs/pointgroup_ops
