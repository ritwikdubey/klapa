

class LSTM(RNN):
	"""Long Short Term Memory Layer
	Reference:
	    - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
    			activation='tanh',
    			recurrent_activation='hard_sigmoid',
    			use_bias=True,
    			kernel_initializer='glorot_uniform',
    			recurrent_initializer='orthogonal',
    			bias_initializer='zeros',
    			unit_forget_bias=True,
    			kernel_regularizer=None,
    			recurrent_regularizer=None,
    			bias_regularizer=None,
    			activity_regularizer=None,
    			kernel_constraint=None,
    			recurrent_contraint=None,
    			bias_constraint=None,
    			dropout=0,
    			recurrent_dropout=0,
    			implementation=1,
    			**kwargs):
    	if implementation == 0:
    		warnings.warn('implementation=0 has been ')
    	if K.backend() == 'cntk':
    		if not kwargs.get('unroll') and (dropout > 0 or recurrent_dropout > 0):
    			warnings.warn('RNN dropout is not supported with the CNTK backend')
    			dropout = 0
    			recurrent_dropout = 0

    	cell = LSTMCell(units,
    					activation=activation,
    					recurrent_activation=recurrent_activation,
    					use_bias=use_bias,
		    			kernel_initializer=kernel_initializer,
		    			recurrent_initializer=recurrent_initializer,
		    			bias_initializer=bias_initializer,
		    			unit_forget_bias=unit_forget_bias,
		    			kernel_regularizer=kernel_regularizer,
		    			recurrent_regularizer=recurrent_regularizer,
		    			bias_regularizer=bias_regularizer,
		    			activity_regularizer=activity_regularizer,
		    			kernel_constraint=kernel_constraint,
		    			recurrent_contraint=recurrent_contraint,
		    			bias_constraint=bias_constraint,
		    			dropout=dropout,
		    			recurrent_dropout=recurrent_dropout,
		    			implementation=implementation)
    	super(LSTM, self).__init__(cell, **kwargs)
    	self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
    	self.cell._generate_dropout_mask(inputs, training=training)
    	self.cell._generate_recurrent_dropout_mask(inputs, training=training)
    	return super(LSTM, self).call(inputs,
    								  mask=mask,
    								  training=training,
    								  initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
    	config = {'units': self.units,
    				'activation': activations.serialize(self.activation),
    				'recurrent_activation': activations.serialize(self.recurrent_activation),
    				'use_bias': self.use_bias,
    				'kernel_initializer': initializers.serialize(self.kernel_initializer),
    				'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
    				'bias_initializer': initializers.serialize(self.bias_initializer),
    				'unit_forget_bias': self.unit_forget_bias,
    				'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
    				'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
    				'bias_regularizer': regularizers.serialize(self.bias_regularizer),
    				'activity_regularizer': regularizers.serialize(self.activity_regularizer),
    				'kernel_constraint': constraints.serialize(self.kernel_constraint),
    				'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
    				'bias_constraint': constraints.serialize(self.bias_constraint),
    				'dropout': self.dropout,
    				'recurrent_dropout': self.recurrent_dropout,
    				'implementation': self.implementation}
    	base_config = super(LSTM, self).get_config()
    	del base_config['cell']
    	return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
    	if 'implementation' in config and config['implementation'] == 0:
    		config['implementation'] = 1
    	return cls(**config)

