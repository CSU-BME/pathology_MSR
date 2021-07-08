from ISR.models import RRDN_ADV
from ISR.models import Discriminator
from ISR.models import Cut_VGG19

lr_train_patch_size = 64
layers_to_extract = [5, 9]
scale = 8 #[2,4,8,16.....]
hr_train_patch_size = lr_train_patch_size * scale

rrdn  = RRDN_ADV(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=None, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

from ISR.train import Trainer_adv
loss_weights = {
  'generator': 0.06,
  'feature_extractor': 0.083,
  'discriminator': 0.04
}
losses = {
  'generator': 'mae',
  'feature_extractor': 'mse',
  'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}#5

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

trainer.train(
    epochs=120,
    steps_per_epoch=500,
    batch_size=2,
    monitored_metrics={'val_generator_PSNR_Y': 'max'}
)

#discriminatore
def train_discriminatore_in_combined_model(self):
    combined = self.trainer._combine_networks()
    self.assertTrue(combined.get_layer('discriminator').trainable == True)
    self.assertTrue(combined.get_layer('feature_extractor').trainable == True)
