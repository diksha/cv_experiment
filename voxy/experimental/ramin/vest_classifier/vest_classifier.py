#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#


#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


# In[2]:


#hide
from fastbook import *
from fastai.callback.wandb import *
from fastai.vision.all import *


# In[3]:


import wandb
arch_str = 'resnet18'
wandb.init(project='vest_classifier_2', name=arch_str)


# In[4]:


path = Path('/home/ramin_voxelsafety_com/data/vest_patches')


# In[5]:


#hide
Path.BASE_PATH = path


# In[8]:


path.ls()


# In[9]:


bs = 64
arch = models.resnet18
arch_str = 'resnet18'


# In[10]:


def get_label(fname):
  return str(fname.parent).split('/')[-1]


# In[37]:


big_size = int(460 /2)
img_size = int(224 / 2)
data = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    get_y=get_label,
    splitter=RandomSplitter(seed=42),  
    item_tfms=Resize(big_size),    
    batch_tfms=aug_transforms(size=img_size, min_scale=0.75) + [Normalize.from_stats(*imagenet_stats)]
)


# In[38]:


dls = data.dataloaders(path, bs=bs)


# In[39]:


len(dls.train.dataset), len(dls.valid.dataset)


# In[40]:


model_cb = SaveModelCallback(monitor='valid_loss', every_epoch=False)
es_cb = EarlyStoppingCallback(monitor='valid_loss', patience=5)
callbacks = [model_cb, es_cb, WandbCallback()]

learner = cnn_learner(dls, arch, metrics=accuracy, concat_pool=False, cbs=callbacks)


# In[41]:


learner.lr_find()


# In[42]:


learner.fine_tune(4, base_lr=5.0e-3)


# In[43]:


save_folder = path.parent/'../vest/vest_models'

os.makedirs(save_folder, exist_ok=true)
save_folder


# In[44]:


model_path = save_folder/f'vest_classifier_model_{arch_str}.pkl'
learner.export(fname=model_path)


# In[45]:


interp = ClassificationInterpretation.from_learner(learner, dl=dls.valid)


# In[46]:


interp.plot_confusion_matrix(figsize=(8,8))


# In[22]:


interp.most_confused(min_val=5)


# In[23]:


interp.print_classification_report()


# In[29]:


interp.plot_top_losses(15, figsize=(30,30))


# In[30]:


dls.train.dataset


# In[47]:


learn = load_learner(model_path, cpu=True)


# In[48]:


final_model = learn.model
final_model.eval();


# In[49]:


example_input = torch.randn(1, 3, img_size, img_size, requires_grad=False).cpu()
traced_model = torch.jit.trace(final_model, example_input)
traced_model.save(save_folder/f'vest_classifier_{arch_str}_traced_model.pth')


# In[34]:


imagenet_stats


# In[35]:


vocab = learn.dls.vocab
vocab

