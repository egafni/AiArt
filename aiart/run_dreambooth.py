# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
import json
import os
import shlex

import dreambooth

DATA_DIR = '/home/ubuntu/projects/AiArt/data'
EXPERIMENT_NAME = 'v5_style_pp_1__te_0'
OUTPUT_DIR = f'/home/ubuntu/experiments/{EXPERIMENT_NAME}'

concepts_list = [
    {
        "instance_prompt": "3dCartoonArt style",
        "class_prompt": "style",
        "instance_data_dir": f"{DATA_DIR}/v2_280",
        "class_data_dir": f"{OUTPUT_DIR}/class_data"
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

# concepts_list[0]['instance_data_dir']

NUM_INSTANCE_IMAGES = 280
NUM_CLASS_IMAGES = min(NUM_INSTANCE_IMAGES * 12, 200)
MAX_NUM_STEPS = NUM_INSTANCE_IMAGES * 80
# LR_SCHEDULE = 'constant'
LR_WARMUP_STEPS = int(MAX_NUM_STEPS / 10)  # note ignoring this
# print(LR_WARMUP_STEPS)
#
# MAX_NUM_STEPS = 960
# NUM_CLASS_IMAGES = 100
# MAX_NUM_STEPS = 4000

# MAX_NUM_STEPS = 2000

lr_scheduler = 'constant'
lr_scheduler = 'polynomial'
save_interval = 500

#   --use_8bit_adam \
# --train_text_encoder \
#
cmd = f"""--pretrained_model_name_or_path={MODEL_NAME} \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir={OUTPUT_DIR} \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="{lr_scheduler}" \
  --lr_warmup_steps={LR_WARMUP_STEPS} \
  --num_class_images={NUM_CLASS_IMAGES} \
  --sample_batch_size=4 \
  --max_train_steps={MAX_NUM_STEPS} \
  --save_min_steps=300 \
  --save_interval={save_interval} \
  --save_sample_prompt="xcourses" \
  --concepts_list="concepts_list.json" 
"""
print(cmd)
args = dreambooth.parse_args(shlex.split(cmd))
dreambooth.main(args)
