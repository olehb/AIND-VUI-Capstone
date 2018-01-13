from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    InputLayer, MaxPool1D, Dropout, LeakyReLU)


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
                       pool_size=2, dropout_rate=0.2, lrelu_alpha=0.3):
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
    for i in range(num_layers-1):
        cnn = BatchNormalization()(cnn)
        cnn = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     dilation_rate=(2,),
                     activation=None)(cnn)
        cnn = LeakyReLU(lrelu_alpha)(cnn)
        cnn = Dropout(rate=dropout_rate)(cnn)

    # Do not add batch normalization to the last layer
    cnn = MaxPool1D(pool_size=pool_size)(cnn)

    # Stack of GRU layers
    gru = GRU(units, activation=None, return_sequences=True, implementation=2)(cnn)
    gru = LeakyReLU(lrelu_alpha)(gru)
    for i in range(num_layers-2):
        gru = BatchNormalization()(gru)
        gru = GRU(units, activation=None, return_sequences=True, implementation=2)(gru)
        gru = Dropout(rate=dropout_rate)(gru)

    last_gru = GRU(units, activation=None, return_sequences=True, implementation=2)

    # Adding Bidirectional layer
    simp_rnn = SimpleRNN(units, activation=None)
    bidir_rnn = Bidirectional(last_gru, merge_mode='concat')(gru)

    # Finally adding TimeDistributed
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    y_pred = Activation('softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)

    def calc_output_length(input_length):
        output_length = input_length
        for i in range(num_layers):
            output_length = cnn_output_length(output_length, kernel_size,
                                              "valid", conv_stride, dilation=2)

        # recalculate the length as we are adding pooling layer
        return output_length / pool_size

    model.output_length = calc_output_length

    print(model.summary())

    return model


def final_model():
    """ Build a deep network for speech 
    """
    print(model.summary())
    return model


def __make_gru_layer(prev_layer, units, activation):
    layer = GRU(units, activation=activation,
                return_sequences=True, implementation=2)(prev_layer)
    layer = BatchNormalization()(layer)
    return layer


__make_rnn_layer = __make_gru_layer
