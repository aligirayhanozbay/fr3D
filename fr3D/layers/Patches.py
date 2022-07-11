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
    def __init__(self, projection_dim, individual_embeddings=False):
        super().__init__()
        self.individual_embeddings = individual_embeddings
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
        if not self.individual_embeddings:
            return encoding
        else:
            return encoding, patch_embedding, positional_embedding

class MaskedPatchEncoder(PatchEncoder):
    def __init__(self, projection_dim, mask_proportion, downstream=False, **kwargs):
        super().__init__(projection_dim, individual_embeddings=True, **kwargs)

        if isinstance(mask_proportion, float):
            mask_proportion = (mask_proportion, mask_proportion)
        self.mask_proportion = mask_proportion
        self.downstream = downstream

    def build(self, input_shape):
        super().build(input_shape)
        print('asd')
        self.mask_token = self.add_weight(name='mask_token',
                                          shape=[1,input_shape[2]],
                                          dtype=self.projection.dtype,
                                          initializer=tf.keras.initializers.RandomNormal(0.0,1.0),
                                          trainable=True)

    def get_random_indices(self, batch_size):

        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        num_mask = tf.cast(self.num_patches*tf.random.uniform([],minval=self.mask_proportion[0], maxval=self.mask_proportion[1]), tf.int32)
        mask_indices = rand_indices[:,:num_mask]
        unmask_indices = rand_indices[:num_mask:]
        return mask_indices, unmask_indices

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        patch_embeddings, _, pos_embeddings = super().call(patches)
        pos_embeddings = tf.tile(
            pos_embeddings[tf.newaxis], [batch_size, 1, 1]
        )

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            num_mask = tf.shape(mask_indices)[1]
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

if __name__ == '__main__':
    x = tf.keras.layers.Input(shape=(64,64,64,3))
    o = Patches(8)(x)
    o = MaskedPatchEncoder(768, (0.25,0.50))(o)
    mod = tf.keras.Model(x,o)
    mod.summary()
