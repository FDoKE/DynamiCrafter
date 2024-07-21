
# How to train lora
## Prepare conda evironment (easy way)

### 1. Create environment:

`conda create --name train_lora python==3.10.14`

`conda activate train_lora`

### 2. Install torch

`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

### 3. Install requirements.txt

`pip install -r requirements.txt`

### 4. Download base model (from original repo)
Download ckpt from:
https://huggingface.co/Doubiiu/DynamiCrafter_512

Put it inside `checkpoints/dynamicrafter_512_v1/model.ckpt`

### 5. Prepare train dataset (i added example in repo)

train data in `train` folder.
`data.csv` should be formatted as WebVid dataset (https://huggingface.co/datasets/Doubiiu/webvid10m_motion)

Only theese columns must be filled:
- `videoid` - name of video file
- `page_dir` - folder name inside `videos` where video is placed
- `name` - actual captioning that used during training

_duration and other columns - not used (but fill them anyways as in my repo example)_

### 6. Start training (from main project directory)

`sh configs/training_512_lora_v1.0/run.sh`

# Information about configuration

## Main config file is `configs/configs.training_512_lora_v1.0/config.yaml`
### Look at the `#CONFIGURE` comments

- `lora.rank` - rank of the LoRA training
- `lora.alpha` - alpha value of the lora (0 means no training is done, 1 - lora is fully affecting weights)
-  `trainer.accumulate_grad_batches` - how much iterations will be per gradient step (more, model will better understand training data, like u had batch_size = accumulate_grad_batches), but trains slower
- `callbacks.model_checkpoint.every_n_train_steps` - how frequently last checkpoint will be saved (it rewrites every callback).
  Step calculated as iterations/accumulate_grad_batches, so in case of training data = 200 (as in repo), 200 iterations / 8 = 25 steps, so in repo we save every 20 step.
- `callbacks.metrics_over_trainsteps_checkpoint.every_n_epochs` - defines how ofter checkpoint saved (old don't remove, so don't put low values in case of steps, or small epoch)

### also look at the `# CHANGED` comments:

i made fps of videos fixed and sample with stride = 1, you can change it and make it as you want or return to defaults of the dynami crafter:
```
video_length: 16
frame_stride: 6
load_raw_resolution: true
resolution: [320, 512]
spatial_transform: resize_center_crop
random_fs: true  ## if true, we uniformly sample fs with max_fs=frame_stride (above)
```

Most of the parameters are known. Feel free to google them.



# Extracting and applying lora

## Extract lora:
`python extract_lora.py --model=trained/training_512_lora_v1.0/checkpoints/epoch=0-step=20.ckpt `

it will produce lora weights in .safetensors format

## Apply lora to a base model:

`python apply_lora.py --base_model=checkpoints/dynamicrafter_512_v1/model.ckpt --lora=epoch=0-step=20_lora.safetensors --alpha=1.0`

## Now you can use that model for inference good luck! :)
## I use https://github.com/kijai/ComfyUI-DynamiCrafterWrapper in comfy
## If you have any questions, you can freely reach me