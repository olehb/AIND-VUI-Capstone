from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    InputLayer, MaxPool1D, Dropout, LeakyReLU, GaussianNoise)


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    bn_rnn = __make_rnn_layer(input_data, units, activation)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
                         return_sequences=True, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29, activation='relu'):
    """ Build a deep recurrent network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    rnn = __make_rnn_layer(input_data, units, activation)
    for i in range(recur_layers-1):
        rnn = __make_rnn_layer(rnn, units, activation)

    time_dense = TimeDistributed(Dense(output_dim))(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def deep_rnn_model_seq(input_dim, units, recur_layers, output_dim=29, activation='relu'):
    """ Build a deep recurrent network for speech
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(None, input_dim)))
    model.add(GRU(units, activation=activation,
                  return_sequences=True, implementation=2))
    model.add(BatchNormalization())
    for i in range(recur_layers-1):
        model.add(GRU(units, activation=activation,
                      return_sequences=True, implementation=2))
        model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation(K.softmax))
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29, activation='relu'):
    """ Build a bidirectional recurrent network for speech
    """
    input_data = Input(name='the_input', shape=(None, input_dim))
    rnn = GRU(units, activation=activation,
              return_sequences=True, implementation=2)
    bidir_rnn = Bidirectional(rnn, merge_mode='concat')(input_data)
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def deep_cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                       conv_border_mode, units, num_layers=2, output_dim=29,
                       pool_size=2, dropout_rate=0.2, lrelu_alpha=0.3,
                       dilation_rate=(2,)):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Stack of Convolutional layers
    cnn = Conv1D(filters, kernel_size,
                  strides=conv_stride,
                  padding=conv_border_mode,
                  activation=None)(input_data)
    cnn = LeakyReLU(lrelu_alpha)(cnn)
    for _ in range(num_layers-1):
        cnn = BatchNormalization()(cnn)
        cnn = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     dilation_rate=dilation_rate,
                     activation=None)(cnn)
        cnn = LeakyReLU(lrelu_alpha)(cnn)
        cnn = Dropout(rate=dropout_rate)(cnn)

    # Do not add batch normalization to the last layer
    cnn = MaxPool1D(pool_size=pool_size)(cnn)

    # Stack of GRU layers
    gru = cnn
    for _ in range(num_layers-2):
        gru = GRU(units, activation=None, return_sequences=True, implementation=2)(gru)
        gru = LeakyReLU(lrelu_alpha)(gru)
        gru = BatchNormalization()(gru)
        gru = Dropout(rate=dropout_rate)(gru)

    last_gru = GRU(units, activation=None, return_sequences=True, implementation=2)

    # Adding Bidirectional layer
    bidir_rnn = Bidirectional(last_gru, merge_mode='concat')(gru)

    # Finally adding TimeDistributed
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    y_pred = Activation('softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)

    def calc_output_length(input_length):
        output_length = input_length
        for _ in range(num_layers):
            output_length = cnn_output_length(output_length, kernel_size,
                                              conv_border_mode, conv_stride,
                                              dilation=dilation_rate[0])
        return output_length / pool_size

    model.output_length = calc_output_length

    print(model.summary())

    return model


def deep_cnn_cudnngru_model(input_dim, filters, kernel_size, conv_stride,
                            conv_border_mode, units, num_layers=2, output_dim=29,
                            pool_size=2, dropout_rate=0.2, lrelu_alpha=0.3,
                            dilation_rate=(2,)):
    """ Build a recurrent + convolutional network for speech
    """
    # Importing here since it's only supported in newest keras
    from keras.layers import CuDNNGRU

    input_data = Input(name="the_input", shape=(None, input_dim))
    cnn = Conv1D(filters, kernel_size,
                 strides=conv_stride,
                 padding=conv_border_mode,
                 dilation_rate=dilation_rate,
                 activation=None
                 )(input_data)

    for _ in range(num_layers-1):
        cnn = LeakyReLU(lrelu_alpha)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        cnn = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation=None)(cnn)

    # Since we're using MaxPool, LeakyRelu step is redundand
    # Do not add batch normalization to the last layer
    # Adding MaxPool layer on top of the stack of convolutions
    max_pool = MaxPool1D(pool_size=pool_size)(cnn)

    # Stack of GRU layers
    gru = max_pool
    for _ in range(num_layers-1):
        gru = CuDNNGRU(units, return_sequences=True)(gru)
        gru = LeakyReLU(lrelu_alpha)(gru)
        gru = BatchNormalization()(gru)
        gru = Dropout(dropout_rate)(gru)

    # Adding Bidirectional layer
    bidir = Bidirectional(CuDNNGRU(units, return_sequences=True))(gru)

    # TODO: Activation after Bidirectional?

    # Finally adding TimeDistributed
    time_dist = TimeDistributed(Dense(output_dim))(bidir)
    y_pred = Activation(K.softmax)(time_dist)
    model = Model(inputs=input_data, outputs=y_pred)

    def calc_output_length(input_length):
        output_length = input_length
        for _ in range(num_layers):
            output_length = cnn_output_length(output_length, kernel_size,
                                              conv_border_mode, conv_stride,
                                              dilation=dilation_rate[0])
        return output_length//pool_size
    model.output_length = calc_output_length

    print(model.summary())

    return model


def deep_cnn_cudnnlstm_model(input_dim, filters, kernel_size, conv_stride,
                             conv_border_mode, units, num_layers=2, output_dim=29,
                             pool_size=2, dropout_rate=0.2, lrelu_alpha=0.3,
                             dilation_rate=(2,)):
    """ Build a recurrent + convolutional network for speech
    """
    # Importing here since it's only supported in newest keras
    from keras.layers import CuDNNLSTM

    input_data = Input(name="the_input", shape=(None, input_dim))
    cnn = Conv1D(filters, kernel_size,
                 strides=conv_stride,
                 padding=conv_border_mode,
                 dilation_rate=dilation_rate,
                 activation=None
                 )(input_data)

    for _ in range(num_layers-1):
        cnn = LeakyReLU(lrelu_alpha)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        cnn = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation=None)(cnn)

    # Since we're using MaxPool, LeakyRelu step is redundand
    # Do not add batch normalization to the last layer
    # Adding MaxPool layer on top of the stack of convolutions
    max_pool = MaxPool1D(pool_size=pool_size)(cnn)

    # Stack of GRU layers
    gru = max_pool
    for _ in range(num_layers-1):
        gru = Bidirectional(CuDNNLSTM(units, return_sequences=True))(gru)
        gru = LeakyReLU(lrelu_alpha)(gru)
        gru = BatchNormalization()(gru)
        gru = Dropout(dropout_rate)(gru)

    # Finally adding TimeDistributed
    time_dist = TimeDistributed(Dense(output_dim))(gru)
    y_pred = Activation(K.softmax)(time_dist)
    model = Model(inputs=input_data, outputs=y_pred)

    def calc_output_length(input_length):
        output_length = input_length
        for _ in range(num_layers):
            output_length = cnn_output_length(output_length, kernel_size,
                                              conv_border_mode, conv_stride,
                                              dilation=dilation_rate[0])
        return output_length//pool_size
    model.output_length = calc_output_length

    print(model.summary())

    return model


#######################################################################################
### Gaussian Noise and separate params for conv and rnn layers
#######################################################################################
def deep_cnn_cudnngru_model_2(input_dim, filters, kernel_size, conv_stride,
                            units, conv_border_mode='valid', conv_layers=2, rnn_layers=2,
                            output_dim=29, pool_size=3, dropout_rate=0.2,
                            lrelu_alpha=0.3, dilation_rate=(2,), noise=0.1):
    """ Build a recurrent + convolutional network for speech
    """
    # Importing here since it's only supported in newest keras
    from keras.layers import CuDNNGRU

    input_data = Input(name="the_input", shape=(None, input_dim))
    noise = GaussianNoise(noise)(input_data)
    cnn = Conv1D(filters, kernel_size,
                 strides=conv_stride,
                 padding=conv_border_mode,
                 dilation_rate=dilation_rate,
                 activation=None
                 )(noise)

    for _ in range(conv_layers-1):
        cnn = LeakyReLU(lrelu_alpha)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        cnn = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation=None)(cnn)

    # Since we're using MaxPool, LeakyRelu step is redundand
    # Do not add batch normalization to the last layer
    # Adding MaxPool layer on top of the stack of convolutions
    max_pool = MaxPool1D(pool_size=pool_size)(cnn)

    # Stack of GRU layers
    gru = Bidirectional(CuDNNGRU(units, return_sequences=True))(max_pool)
    for _ in range(rnn_layers-1):
        gru = LeakyReLU(lrelu_alpha)(gru)
        gru = BatchNormalization()(gru)
        gru = Dropout(dropout_rate)(gru)
        gru = Bidirectional(CuDNNGRU(units, return_sequences=True))(gru)

    # Finally adding TimeDistributed
    time_dist = TimeDistributed(Dense(output_dim))(gru)
    y_pred = Activation(K.softmax)(time_dist)
    model = Model(inputs=input_data, outputs=y_pred)

    def calc_output_length(input_length):
        output_length = input_length
        for _ in range(conv_layers):
            output_length = cnn_output_length(output_length, kernel_size,
                                              conv_border_mode, conv_stride,
                                              dilation=dilation_rate[0])
        return output_length//pool_size
    model.output_length = calc_output_length

    print(model.summary())

    return model


def final_model():
    """ Build a deep network for speech 
    """
    model = deep_cnn_cudnngru_model(input_dim=13, # change to 13 if you would like to use MFCC features
                                    filters=200,
                                    kernel_size=11, 
                                    conv_stride=1,
                                    conv_border_mode='valid',
                                    units=200)
    print(model.summary())
    return model


def __make_gru_layer(prev_layer, units, activation):
    layer = GRU(units, activation=activation,
                return_sequences=True, implementation=2)(prev_layer)
    layer = BatchNormalization()(layer)
    return layer


__make_rnn_layer = __make_gru_layer
