import tensorflow as tf

from .ConvAutoencoder import decoder_block, ConvAutoencoderC

def ConvDecoder(input_units: int, decoder_trainable=True, activation=tf.nn.leaky_relu, decoder_input_shape = None, **decoder_args):

    if 'decoder' in decoder_args:
        decoder = decoder_args['decoder']
    else:
        decoder = decoder_block(input_tensor=decoder_input_shape,
                                activation=activation,
                                **decoder_args)
    decoder.trainable= decoder_trainable

    latent_space_shape = decoder_block.input_shape[1:]
    latent_space_embedder = ConvAutoencoderC.make_latent_space_embedder(latent_space_shape,
                                                                        input_units)


    inp = tf.keras.layers.Input(shape=(input_units,))
    x = latent_space_embedder(inp)
    x = decoder(x)

    model = tf.keras.Model(inp, x)
        
    
                
