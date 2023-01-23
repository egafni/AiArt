# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
import json
import os
import shlex

import dreambooth

concepts_list = [
    {
        "instance_prompt": "xcourses",
        "class_prompt": "portrait photo of a man",
        "instance_data_dir": "/home/ubuntu/data/inputs",
        "class_data_dir": "/home/ubuntu/data/class_data"
        # `class_data_dir` contains regularization images
    },
    #     {
    #         "instance_prompt":      "photo of ukj person",
    #         "class_prompt":         "photo of a person",
    #         "instance_data_dir":    "/content/data/ukj",
    #         "class_data_dir":       "/content/data/person"
    #     }
]

for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)

MODEL_NAME = 'runwayml/stable-diffusion-v1-5'
EXPERIMENT_NAME = 'v1'
OUTPUT_DIR = f'/home/ubuntu/experiments/{EXPERIMENT_NAME}'

NUM_INSTANCE_IMAGES = 12
NUM_CLASS_IMAGES = NUM_INSTANCE_IMAGES * 12
MAX_NUM_STEPS = NUM_INSTANCE_IMAGES * 80 * 2
# MAX_NUM_STEPS = 50
LR_SCHEDULE = 'constant'
LR_WARMUP_STEPS = int(MAX_NUM_STEPS / 10)  # note ignoring this
print(LR_WARMUP_STEPS)

MAX_NUM_STEPS = 960

cmd = f"""--pretrained_model_name_or_path={MODEL_NAME} \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir={OUTPUT_DIR} \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images={NUM_CLASS_IMAGES} \
  --sample_batch_size=4 \
  --max_train_steps={MAX_NUM_STEPS} \
  --save_min_steps=800 \
  --save_interval=100 \
  --save_sample_prompt="xcourses" \
  --concepts_list="concepts_list.json" 
"""
print(cmd)
args = dreambooth.parse_args(shlex.split(cmd))
dreambooth.main(args)
