# WARNING only tested with python 3.8
pip install -r requirements.txt
pip install --no-deps https://github.com/brian6091/xformers-wheels/releases/download/0.0.15.dev0%2B4c06c79/xformers-0.0.15.dev0+4c06c79.d20221205-cp38-cp38-linux_x86_64.whl
#pip install git+https://github.com/ShivamShrirao/diffusers
pip install diffusers
pip install -U --pre triton
pip install accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort
