import numpy as np
from PIL import Image
from ISR.models import RRDN_ADV
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.predict import Predictor_ext_adv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


#input_path='/media/disk2/group3_SR/sample_128_ori'
input_path='./sample/'
output_path=input_path+'_output4'


lr_patch_size = 64
layers_to_extract = [5, 9]
scale = 8
hr_train_patch_size = lr_patch_size * scale
rrdn  = RRDN_ADV(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_patch_size)
#f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
#discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

predictor=Predictor_ext_adv(input_path,output_path)
predictor.get_predictions(rrdn,'./weights/rrdn-C4-D3-G64-G064-T10-x8/sample_weights/rrdn-C4-D3-G64-G064-T10-x8_best-val_generator_PSNR_Y_epoch115.hdf5'
                          ,patch_size=lr_patch_size)