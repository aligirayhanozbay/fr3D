#based on https://keras.io/examples/vision/masked_image_modeling/
import tensorflow as tf

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def create_3d_patches(self, images):
        patches = tf.extract_volume_patches(images,
                                            ksizes=self.ksizes,
                                            strides=self.strides,
                                            padding="VALID")
        return patches

    def create_2d_patches(self, images):
        patches = tf.image.extract_patches(images,
                                           sizes=self.ksizes,
                                           strides=self.strides,
                                           rates=[1,1,1,1],
                                           padding="VALID")
        return patches

    def transpose(self, images):
        if self.ndims == 1:
            return tf.transpose(images,[0,2,1])
        elif self.ndims == 2:
            return tf.transpose(images,[0,2,3,1])
        elif self.ndims == 3:
            return tf.transpose(images,[0,2,3,4,1])

    def build(self, input_shape):
        if len(input_shape) == 5:
            self.ndims = 3
            self.create_patches = self.create_3d_patches
        elif len(input_shape) == 4:
            self.ndims = 2
            self.create_patches = self.create_2d_patches

        self.n_channels = input_shape[1 if tf.keras.backend.image_data_format() == 'channels_first' else -1]

        self.ksizes = [1]+ [self.patch_size]*self.ndims + [1]
        self.strides = self.ksizes

        self.resize = tf.keras.layers.Reshape((-1, self.n_channels*(self.patch_size**self.ndims)))
            

    def get_config(self):
        return {"patch_size": self.patch_size}

    def call(self, images):
        if tf.keras.backend.image_data_format() == 'channels_first':
            images = self.transpose(images)
            
        patches = self.create_patches(images)
        patches = self.resize(patches)
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, projection_dim):
        super().__init__()
        self.projection = tf.keras.layers.Dense(projection_dim)

    def build(self, input_shape):
        self.num_patches = input_shape[1]
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.num_patches, output_dim=self.projection.units)
        self.projection.build(input_shape)

    def get_config(self):
        return {"projection_dim": self.projection.units}

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        patch_embedding = self.projection(patches)
        positional_embedding = self.position_embedding(positions)
        encoding = patch_embedding + positional_embedding
        return encoding

class MaskedPatchEncoder(PatchEncoder):
    def __init__(self, projection_dim, mask_proportion, downstream=False, **kwargs):
        super().__init__(projection_dim, **kwargs)

        if isinstance(mask_proportion, float):
            mask_proportion = (mask_proportion, mask_proportion)
        self.mask_proportion = mask_proportion
        self.downstream = downstream

    def build(self, input_shape):
        super().build(input_shape)
        self.mask_token = self.add_weight(name='mask_token',
                                          shape=[1,1,self.projection.units],
                                          dtype=self.projection.dtype,
                                          initializer=tf.keras.initializers.RandomNormal(0.0,1.0),
                                          trainable=True)

    def get_mask(self, batch_size):
        proportion_masked = tf.random.uniform(shape=(batch_size,1), minval=self.mask_proportion[0], maxval=self.mask_proportion[1])
        randvals = tf.random.uniform(shape=(batch_size, self.num_patches))
        return tf.cast(randvals <= proportion_masked, tf.keras.backend.floatx())

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        encodings = super().call(patches)
        
        mask = self.get_mask(batch_size)
        
        return  (1-mask[...,tf.newaxis])*encodings + mask[...,tf.newaxis]*self.mask_token
        

if __name__ == '__main__':
    z = tf.random.uniform((2,64,64,64,3))
    x = tf.keras.layers.Input(shape=(64,64,64,3))
    o = Patches(8)(x)
    o = MaskedPatchEncoder(768, (0.25,0.50))(o)
    mod = tf.keras.Model(x,o)
    mod.compile(loss='mse', optimizer='sgd')
    mod.summary()
    zz = mod(z)
