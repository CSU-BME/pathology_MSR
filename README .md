# Multi-scale super-resolution generation of low-resolution scanned pathological images
This is the implementation of the models and source code for the " Multi-scale super-resolution generation of low-resolution scanned pathological images"。

# File description
- train_adv.py: code to train our generator with loss,train_feature_extractor loss,inlucde configuration for file paths and hyper parameters for networks.
- predict_adv:code to test our model.
- rrdn_adv:gans,it can carry out continuous multi-stage magnification, from 2 times to 4 times to 8 times, of which 4 times magnification is the best.
- weights: weight model trained by train_adv.py 

#data processing
 Since we are using pathology images data, before training, we have to crop the hr data into 128size lr pictures. Of course, you can also use DIK2 data for training.
 - Download DIV2K dataset from https://data.vision.ee.ethz.ch/cvl/DIV2K/
 - Set train path and valid paths for the augmented dataset and validation set in train_adv.py

# Usage for training

Create the models
```python
from ISR.models import RRDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19

lr_train_patch_size = 64
layers_to_extract = [5, 9]
scale = 8
hr_train_patch_size = lr_train_patch_size * scale

rrdn = RRDN_ADV(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=None, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)
```

Create a Trainer object using the desired settings and give it the models (`gener`and'`f_ext` and `discr` are optional)
```python
from ISR.train import trainer_adv
loss_weights = {
  'generator': 0.06,
  'feature_extractor': 0.083,
  'discriminator': 0.01
}
losses = {
  'generator': 'mae',
  'feature_extractor': 'mse',
  'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer_adv(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='./data/lr128_data',
    hr_train_dir=['/data/hr256_data','/data/hr512_data','/data/hr1024_data'],
    lr_valid_dir='./data/lr128_val',
    hr_valid_dir=['/data/hr256_val','/data/hr512_val','/data/hr1024_val'],
   loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='image_dataset',
    log_dirs=log_dirs,
    #weights_generator='rrdn-C4-D3-G64-G064-T10-x8_best-val_generator_PSNR_Y_epoch109.hdf5/',
    weights_generator=None,
    weights_discriminator=None,
    n_validation=40,
)
```

Start training(`epochs`and'`per_epochs` and `batch_size` are optional)
```python
trainer.train(
    epochs=120,
    steps_per_epoch=500,
    batch_size=2,
    monitored_metrics={'val_PSNR_Y': 'max'}
)
```
# Usage for testing
- Set test paths for test set and trained models in predict_adv.py (the trained models put in weights folder and must chose the best rrdn models)
- Run predict_adv.py

# Acknowledgement
- Code architecture is based on [Image Super-Resolution (ISR)](https://github.com/idealo/image-super-resolution/)

