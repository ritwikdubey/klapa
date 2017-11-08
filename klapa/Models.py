# Models.py
from __future__ import absolute_import
from __future__ import print_function

import warnings
import copy
import json
import yaml
import numpy as np

from . import backend as K
from . import optimizers
from . import layers as layer_module
from .utils.io_utils import ask_to_proceed_with_overwrite
from .utils.generic_utils import has_arg
from .engine.training import Model
from .engine import topology
from .legacy import layers as legacy_layers
from .legacy import models as legacy_models
from .legacy import interfaces

try:
    import h5py
except ImportError:
    hspy = None

class Sequential(Model):
    """Linear stack of layers. THe first layer passed to a sequential model should have defined
    input shape. It should receive an input named input_shape or batch_input_shape argument, or
    for some type of layer an input_dim argument.
    """
    def __init__(self, layers=None, name=None):
        self.layers = []
        self.model = None # internal model instance
        self.inputs = [] # list of input tensors
        self.outputs = [] # list of output tensors
        self._trainable = True
        self._initial_weights = None

        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False

        if not name:
            prefix = 'sequential_'
            name = prefix + str(K.get_uid(prefix))
        self.name = name

        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """Add a layer instance on top of the layer stack
        Exceptions:
        TypeError: if layer is not a layer instance
        ValueError: if layer argument does not know its shape
        ValueError: in case layer argument has multiple output tensors or is already connected
        somewhere else
        """
        if not isinstance(layer, Layer):
            raise TypeError('Added layer must be an instance of class Layer. Found: ' + str(layer))

        if not self.outputs:
            # check for first layer in the model and its an input layer
            if not layer_inbound_nodes:
                # create an input layer
                if not hasattr(layer, 'batch_input_shape'):
                    raise ValueError('The first layer in sequential model must have input_shape or '
                                    'batch_input_shape argument')
                # create the input layer
                x = Input(batch_shape=layer.batch_input_shape, dtype=layer.dtype, name=layer.name + '_input')
                layer(x)

            if len(layer.inbound_nodes) != 1:
                raise ValueError('A layer added to Sequential model must not already be connected'
                                'somewhere else. Model received layer ' + layer.name + ' which has '
                                + str(len(layer.inbound_nodes)) + ' preexisting inbound connections')

            if len(layer.inbound_nodes[0].output_tensors) != 1:
                raise ValueError('All layers in sequential model should have single output tensor.'
                                    'For multi-output layers use the functional API')

            self.outputs = [layer.inbound_nodes[0].output_tensors[0]]
            self.inputs = topology.get_source_inputs(self.outputs[0])

            topology.Node(outbound_layer=self,
                            inbound_layers=[],
                            node_indices=[],
                            tensor_indices=[],
                            input_tensors=self.inputs,
                            output_tensors=self.outputs,
                            input_masks=[None for _ in self.inputs],
                            output_masks=[None],
                            input_shapes=[x._klapa_shape for x in self.inputs],
                            output_shapes=[self.outputs[0]._klapa_shape])
        else:
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError('All layers in sequential model should have single output tensor')
            self.outputs = [output_tensor]
            self.inbound_nodes[0].output_tensors = self.outputs
            self.inbound_nodes[0].output_shapes = [self.outputs[0]._klapa_shape]

        self.layers.append(layer)
        self.built = False

    def pop(self):
        """Remove last layer in the model
        """
        if not self.layers:
            raise TypeError('There are no layers in the model')

        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
            self.inbound_nodes[0].output_tensors = self.outputs
            self.inbound_nodes[0].output_shapes = [self.outputs[0]._klapa_shape]
        self.build = False

    def get_layer(self, name=None, index=None):
        """Retrieve a layer that is part of the model. Returns the layer either based on the name
        or its index in the graph. Indices are based on order of horizontal graph traversal
        """
        if not self.built:
            self.build()
        return self.model.get_layer(name, index)

    def call(self, inputs, mask=None):
        if not self.built:
            self.build()
        return self.model.call(inputs, mask)

    def build(self, input_shape=None):
        if not self.inputs or not self.outputs:
            raise TypeError('Sequential model cannot be built: model is empty')
        self.model = Model(self.inputs, self.outputs[0], name=self.name + '_model')
        self.model.trainable = self.trainable

        # mirror model attributes
        self.supports_masking = self.model.supports_masking
        self._output_mask_cache = self.model._output_mask_cache
        self._output_tensor_cache = self.model._output_tensor_cache
        self._output_shape_cache = self.model._output_shape_cache
        self.input_layers = self.model.input_layers
        self.input_layers_node_indices = self.model.input_layers_node_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_node_indices = self.model.output_layers_node_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self.nodes_by_depth = self.model.nodes_by_depth
        self.container_nodes = self.model.container_nodes
        self.output_names = self.model.output_names
        self.input_names = self.model.input_names
        self._feed_input_names = self.model._feed_input_names
        self._feed_inputs = self.model._feed_inputs

        # Child model callback calls parent sequential model
        self.model.callback_model = self
        self.built = True

    @property
    def uses_learning_phase(self):
        if not self.built:
            self.build()
        return self.model.uses_learning_phase
    
    @property
    def _flattened_layers(self):
        layers = []
        if self.layers:
            if isinstance(self.layers[0], legacy_layers.Merge):
                merge = self.layers[0]
                for layer in merge.layers:
                    if hasattr(layer, '_flattened_layers'):
                        for sublayer in layer._flattened_layers:
                            if sublayer not in layers:
                                layers.append(sublayer)
                    elif hasattr(layer, 'layers'):
                        for sublayer in layer.layers:
                            if sublayer not in layers:
                                layers.append(sublayer)
                    else:
                        if layer not in layers:
                            layers.append(layers)
            else:
                if self.layers[0] not in layers:
                    layers.append(self.layers[0])
            for layer in self.layers[1:]:
                if layer not in layers:
                    layers.append(layer)
        return layers

    def _gather_list_attr(self, attr):
        all_attrs = []
        for layer in self._flattened_layers:
            all_attrs += getattr(layer, attr, [])
        return all_attrs

    @property
    def trainable(self):
        return self._trainable
    
    @trainable.setter
    def trainable(self, value):
        if self.model:
            self.model.trainable = value
        self._trainable = value

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        return self._gather_list_attr('trainable_weights')

    @property
    def non_trainable_weights(self):
        weights = self._gather_list_attr('non_trainable_weights')
        if not self.trainable:
            trainable_weights = self._gather_list_attr('trainable_weights')
            return trainable_weights + weights
        return weights
    
    @property
    def updates(self):
        weigths = self._gather_list_attr('non_trainable_weights')
        if not self.trainable:
            trainable_weights = self._gather_list_attr('trainable_weights')
            return trainable_weights + weights
        return self.model.updates
    
    @property
    def state_updates(self):
        if not self.built:
            self.build()
        return self.model.state_updates
    
    def get_updates_for(self, inputs):
        if not self.built:
            self.build()
        return self.model.get_updates_for(inputs)

    @property
    def losses(self):
        if not self.built:
            self.build()
        return self.model.losses
    
    def get_losses_for(self, inputs):
        if not self.built:
            self.build()
        return self.model.get_losses_for(inputs)

    @property
    def regularizer(self):
        if not self.built:
            self.build()
        return self.model.regularizers
    
    def get_weights(self):
        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
            weights = []
            for layer in layers:
                weights.append(layer.get_weights())
            return weights

        if not self.built:
            self.build()
        return self.model.get_weights()

    def set_weights(self, weights):
        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
            for layer in layers:
                nb_param = len(layer.weights)
                layer.set_weights(weights[:nb_param])
                weights = weights[nb_param:]
        if not self.built:
            self.build()
        self.model.set_weights(weights)

    def load_weights(self, filepath, by_name=False):
        if h5py is None:
            raise ImportError('load_weights requires h5py')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
        else:
            layers = self.layers
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)

        if hasattr(f, 'close'):
            f.close()

    def save_weights(self, filepath, overwrite=True):
        if h5py is None:
            raise ImportError('save weights require h5py')

        if not overwrite and os.path.isfile(filepath):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
        else:
            layers = self.layers

        f = h5py.File(filepath, 'w')
        topology.save_weights_to_hdf5_group(f, layers)
        f.flush()
        f.close()

    def compile(self, optimizers, loss, metrics=None, sample_weight_mode=None, 
                weighted_metrics=None, **kwargs):
        """Configure the learning process
        """
        self.build()
        self.model.compile(optimizers, loss, metrics=metrics, sample_weight_mode=sample_weight_mode,
                            weighted_metrics=weighted_metrics, **kwargs)
        self.optimizers = self.model.optimizers
        self.loss = self.model.loss
        self.total_loss = self.model.total_loss
        self.loss_weight = self.model.loss_weight
        self.metrics = self.model.metrics
        self.weighted_metrics = self.model.weighted_metrics
        self.metrics_tensors = self.model.metrics_tensors
        self.metrics_names = self.model.metrics_names
        self.sample_weight_mode = self.model.sample_weight_mode
        self.sample_weights = self.model.sample_weights
        self.targets = self.model.targets

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0, 
            validation_data=None, shuffle=True, class_weight=None, 
            sample_weight=None, initial_epoch=0, **kwargs):
        """Train the models for a fixed number of epochs
        """
        if 'nb_epoch' in kwargs:
            warnings.warn('The nb_epoch argument in fit has been renamed epochs')
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword argument: ' + str(kwargs))

        if not self.built:
            raise RuntimeError('The model needs to be compiled before being used')
        
        return self.model.fit(x, y,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=callbacks,
                            validation_split=validation_split,
                            validation_data=validation_data,
                            shuffle=shuffle,
                            class_weight=class_weight,
                            sample_weight=sample_weight,
                            initial_epoch=initial_epoch)

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        """Compute the loss on input data, batch by batch
        """
        if not self.built:
            raise RuntimeError('Model needs to be compiled before being used')
        return self.model.evaluate(x, y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight)

    def predict(self, x, batch_size=32, verbose=0):
        """Generates output prediction for the input samples
        """
        if not built:
            self.build()
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def predict_on_batch(self, x):
        """Returns prediction on single batch of samples
        """
        if not self.built:
            self.build()
        return self.model.predict_on_batch(x)

    def train_on_batch(self, x, y, class_weight=None, sample_weight=None):
        """Single gradient update over one batch of samples
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled before using it')
        return self.model.train_on_batch(x, y, sample_weight=sample_weight, class_weight=class_weight)

    def test_on_batch(self, x, y, sample_weight=None):
        """Evaluate the model on single batch of samples
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled before used')
        return self.model.test_on_batch(x, y, sample_weight=sample_weight)

    def predict_proba(self, x, batch_size=32, verbose=1):
        """Generate class probability predictions for the input sample. The input samples
        are processed batch by batch
        """
        preds = self.predict(x, batch_size, verbose)
        if preds.min() < 0 or preds.max() > 1:
            warnings.warn('Network returns invalid probability values. The last layer might not'
                            'predictions into probability')
        return preds

    def predict_classes(self, x, batch_size=32, verbose=1):
        """Generate class predictions for the input samples. The input samples are processed 
        batch by batch
        """
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    @interfaces.legacy_generator_methods_support
    def fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None,
                    validation_data=None, validation_steps=None, class_weight=None, 
                    max_queue_size=10, workers=1, use_multiprocessing=False, initial_epoch=0):
        """Fits model on data generated batch by batch by a python generator. The generator
        is run in parallel to the model, for efficiency. This allows you to do real time data augmentation
        on images on CPU in parallel to training your model on GPU
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled before being used')
        return self.model.fit_generator(generator,
                                        steps_per_epoch,
                                        epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight,
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing,
                                        initial_epoch=initial_epoch)




























