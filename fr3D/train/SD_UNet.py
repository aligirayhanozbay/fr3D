import argparse
import json
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_last')
#tf.keras.mixed_precision.set_global_policy('mixed_float16')


from ..models.SD_UNet import SD_UNet
from ..data import DatasetPipelineBuilder

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--shuffle_size', type=int, default=1201)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

for node_config in config['dataset']['node_configurations']:
    if node_config['nodetype'] == 'HDF5IODataset':
        node_config['filepath'] = args.dataset_path

train_dataset_pipeline = DatasetPipelineBuilder(config['dataset']['node_configurations'])
prepare_dataset_for_training = lambda ds: ds.batch(config['dataset']['batch_size']).shuffle(args.shuffle_size)#.prefetch(config['dataset'].get('prefetch', tf.data.AUTOTUNE))
train_dataset = prepare_dataset_for_training(train_dataset_pipeline.get_node(config['dataset']['training_node']).dataset)
test_dataset = train_dataset #FIX LATER

dummy_input, dummy_output = next(iter(train_dataset))
input_units = dummy_input.shape[-1]
output_channels = dummy_output.shape[-1]
grid_shape = dummy_output.shape[1:-1]
del dummy_input, dummy_output

callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    model = SD_UNet(input_units, grid_shape, out_channels = output_channels, **config['model'])
    model.summary()
    loss_fn = tf.keras.losses.get(config['training']['loss'])
    model.compile(loss=loss_fn, optimizer = tf.keras.optimizers.get(config['training']['optimizer']), metrics = config['training'].get('metrics', None))
    model.fit(train_dataset, epochs = config['training']['epochs'], callbacks = callbacks)#validation_data = test_dataset,
