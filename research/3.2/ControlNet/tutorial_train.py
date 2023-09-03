# Train Control-Net, adapted from https://github.com/lllyasviel/ControlNet/blob/main/tutorial_train.py

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Decide whether to train from start or resume from checkpoint
train_from_start = True
# If not train from start, enter path of ckpt here:
ckpt_path=f"./lightning_logs/version_{9504}/checkpoints/epoch={67}-step={16999}.ckpt"

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
if train_from_start:
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=-1, max_epochs=42, precision=16, callbacks=[logger], accumulate_grad_batches=4) 
#trainer = pl.Trainer(gpus=-1, precision=16, callbacks=[logger], accumulate_grad_batches=4) 

# Train!
if train_from_start:
    trainer.fit(model, dataloader)
else:
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)