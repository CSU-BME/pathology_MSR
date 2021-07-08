import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model
import numpy as np

from ISR.models.imagemodel import ImageModel

WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}

from ISR.utils.image_processing import (
    process_array,
    process_output,
    split_image_into_overlapping_patches,
    stich_together,
)

def make_model(arch_params, patch_size):
    """ Returns the model.

    Used to select the model.
    """
    
    return RRDN(arch_params, patch_size)


def get_network(weights):
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))
    c_dim = 3
    kernel_size = 3
    return arch_params, c_dim, kernel_size, url, name


class RRDN_ADV(ImageModel):
    """Implementation of the Residual in Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1809.00219 (Wang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, T, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        beta: float <= 1, scaling parameter for the residual connections.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs inside each Residual in Residual Dense Block (RRDB).
        T: integer, number or RRDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RRDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(
            self, arch_params={}, patch_size=None, beta=0.2, c_dim=3, kernel_size=3, init_val=0.05, weights=''
    ):
        #if weights:
         #   arch_params, c_dim, kernel_size, url, fname = get_network(weights
        self.params = arch_params
        if self.params['x'] % 2 !=0:
            print('scale must be 2**n')
            return

        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.scale = self.params['x']   #final scale such as 2X, 4X and so on
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_model()
        self.model._name = 'generator'
        self.name = 'rrdn'
        #if weights:
         #   weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
          #  self.model.load_weights(weights_path)
    
    def _dense_block(self, input_layer, d, t):
        """
        Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).

        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)
        """
        
        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = Conv2D(
                self.G,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
            )(x)
            F_dc = Activation('relu')(F_dc)
            x = concatenate([x, F_dc], axis=3)
        
        # DIFFERENCE: in RDN a kernel size of 1 instead of 3 is used here
        x = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
        )(x)
        return x
    
    def _RRDB(self, input_layer, t):
        """Residual in Residual Dense Block.

        t is integer, for naming of RRDB.
        beta is scalar.
        """
        
        # SUGGESTION: MAKE BETA LEARNABLE
        x = input_layer
        
        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = Lambda(lambda x: x * self.beta)(LFF)
            x = Add()([x, LFF_beta])
        x = Lambda(lambda x: x * self.beta)(x)
        x = Add()([input_layer, x])
        return x
    
    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling part. """
        
        x = Conv2D(
            self.c_dim * 2 ** 2,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            #name='PreShuffle',
        )(input_layer)
        return Lambda(
            lambda x: tf.nn.depth_to_space(x, block_size=2, data_format='NHWC'),
            #name='PixelShuffle',
        )(x)

    def _build_rdn(self,pre_blocks):    #for  2X upscale rdn
        # DIFFERENCE: in RDN an extra convolution is present here
        for t in range(1, self.T + 1):
            if t == 1:
                x = self._RRDB(pre_blocks, t)
            else:
                x = self._RRDB(x, t)
        # DIFFERENCE: in RDN a conv with kernel size of 1 after a concat operation is used here
        post_blocks = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            #name='post_blocks_conv',
        )(x)
        # Global Residual Learning
        GRL = Add()([post_blocks, pre_blocks])
        # Upscaling
        PS = self._pixel_shuffle(GRL)
        # Compose SR image
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            #name='SR',
        )(PS)
        return SR

    def _build_model(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR_input')
        mag_num=int(np.log2(self.scale))
        outputs=[]

        for index in range(mag_num):
            if index==0:
                input=LR_input
            else:
                input=outputs[index-1]

            pre_blocks = Conv2D(
                    self.G0,
                    kernel_size=self.kernel_size,
                    padding='same',
                    kernel_initializer=self.initializer,
                    name='Pre_blocks_conv'+str(index),
                )(input)

            outputs.append(self._build_rdn(pre_blocks))

        return Model(inputs=LR_input, outputs=outputs)

    def predict(self, input_image_array, by_patch_of_size=None, batch_size=10, padding_size=2):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into
                patches of the given size.
            padding_size: for large image inference. Padding between the patches.
                Increase the value if there is seamlines.
            batch_size: for large image inferce. Number of patches processed at a time.
                Keep low and increase by_patch_of_size instead.
        Returns:
            sr_img: image output.
        """

        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=padding_size
            )
            # return patches
            mag_num = int(np.log2(self.scale))
            collect={}

            for i in range(0, len(patches), batch_size):
                batch = self.model.predict(patches[i: i + batch_size])
                if i == 0:
                    for j in range(mag_num):
                        collect[j] = batch[j]
                else:
                    for j in range(mag_num):
                        collect[j] = np.append(collect[j], batch[j], axis=0)
            scale=2
            sr_img={}
            for i in range(mag_num):
                padded_size_scaled = tuple(np.multiply(p_shape[0:2], int(np.exp2(i+1)))) + (3,)
                scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], int(np.exp2(i+1)))) + (3,)
                sr_img[i] = stich_together(
                    collect[i],
                    padded_image_shape=padded_size_scaled,
                    target_shape=scaled_image_shape,
                    padding_size=padding_size * int(np.exp2(i+1))
                )

        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]

        for i in range(mag_num):
            sr_img[i] = process_output(sr_img[i])
        return sr_img



