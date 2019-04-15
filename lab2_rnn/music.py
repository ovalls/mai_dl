# 2nd STEP --> create samples and architecture from bach_dataset.npz
# Baseline code by Javier Béjar on WindPrediction
# Bach dataset with 62 chorales, total of 5665 samples with different times,
# converted into npz of (47 songs x 70 events of time x 17 attributes per song)
# Olga Valls
#
from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM, GRU, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import argparse
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def lagged_matrix(data, lag=1, ahead=0):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    matriu = []
    rows = np.size(data, 0)
    columns = np.size(data, 1)
    #print('columns: {}'.format(columns))
    windows = columns - lag - ahead + 1
    #print('windows: {}'.format(windows))

    for r in range(rows):
        for i in range(windows):
            matriu.append(data[r, i:i + lag + ahead])
            #lvect.append(data[r][i:i + lag + ahead])

    pack = np.vstack(matriu)
    print('size pack: {}'.format(pack.shape))
    # print(pack)

    return pack


def generate_dataset(config, ahead=1, data_path=None):
    """
    Generates the dataset for training, test and validation

    :param ahead: number of steps ahead for prediction

    :return:
    """
    #dataset = config['dataset']
    #datanames = config['datanames']
    datasize = config['datasize']
    testsize = config['testsize']
    vars = config['vars']
    lag = config['lag']

    #info = {}

    # numpy array dataset (3D: 47 songs (matrius) x 70 events of time x 15 attributes per song)
    music_data = np.load(data_path + 'bach_dataset.npz')
    data = [music_data[matriu] for matriu in music_data]
    print('matrius dins npz: {}, shape de cada matriu: {}, '.format(len(data), data[0].shape))
    np_data = np.array(data)
    print('np_data shape: {}'.format(np_data.shape))

    # Agafo atribut chord per fer les prediccions, que es el 14 de la 3a dimensió
    data = np_data[:, :, 14]
    print('data shape: {}'.format(data.shape))
    # print(data)

    scaler = StandardScaler()  # NO SÉ SI CAL pq. ja tinc les dades normalitzades a l'npz!!!!!
    data = scaler.fit_transform(data)
    #print('data dp scalar')
    # print(data)

    print('DATA Dim =', data.shape)
    print('datasize: {}'.format(datasize))

    rows_train = data[:datasize, :]
    print('Train Dim (rows_train) = ', rows_train.shape)
    rows_test = data[datasize:datasize + testsize, :]
    print('Test Dim (rows_test) =', rows_test.shape)

    # lag: correlació que hi ha entre les dades -- amplio samples i redueixo events a mida lag+ahead
    train = lagged_matrix(rows_train, lag=lag, ahead=ahead)
    test = lagged_matrix(rows_test, lag=lag, ahead=ahead)

    train_x, train_y = train[:, :lag], train[:, -1:]
    print('train_x shape1: {}'.format(train_x.shape))
    # reshape a 3D amb dimensió 3 a 1
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    print('train_x shape2: {}'.format(train_x.shape))
    print('train_y shape: {}'.format(train_y.shape))

    test_x, test_y = test[:, :lag], test[:, -1:]
    print('test_x shape1: {}'.format(test_x.shape))
    # reshape a 3D amb dimensió 3 a 1
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    print('test_x shape2: {}'.format(test_x.shape))
    print('test_y shape: {}'.format(test_y.shape))

    #print('train_x: {}'.format(train_x))
    #print('train_y: {}'.format(train_y))
    #print('test_x: {}'.format(test_x))
    #print('test_y: {}'.format(test_y))

    return train_x, train_y, test_x, test_y

#    raise NameError('ERROR: No such dataset type')


def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, impl=1):
    """
    RNN architecture

    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
        model.add(BatchNormalization())
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        model.add(BatchNormalization())
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
            model.add(BatchNormalization())
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))
        model.add(BatchNormalization())

    model.add(Dense(1))

    return model


# MAIN

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2

    config = load_config_file(args.config)
    ############################################
    # Data

    ahead = config['data']['ahead']

    # if args.verbose:
    print('-----------------------------------------------------------------------------')
    print('Steps Ahead = %d ' % ahead)

    # Modify conveniently with the path for your data
    aq_data_path = '../datasets/'
    # aq_data_path = 'datasets/'     # local execution

    train_x, train_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, data_path=aq_data_path)

    ############################################
    # Model

    model = architecture(neurons=config['arch']['neurons'],
                         drop=config['arch']['drop'],
                         nlayers=config['arch']['nlayers'],
                         activation=config['arch']['activation'],
                         activation_r=config['arch']['activation_r'], rnntype=config['arch']['rnn'], impl=impl)
    # if args.verbose:
    model.summary()
    print('lag: ', config['data']['lag'],
          '/Neurons: ', config['arch']['neurons'],
          '/Layers: ', config['arch']['nlayers'],
          '/Activations:', config['arch']['activation'], config['arch']['activation_r'])
    print('Tr:', train_x.shape, train_y.shape, 'Ts:', test_x.shape, test_y.shape)
    print()

    ############################################
    # Training

    optimizer = config['training']['optimizer']

    if optimizer == 'rmsprop':
        if 'lrate' in config['training']:
            optimizer = RMSprop(lr=config['training']['lrate'])
        else:
            optimizer = RMSprop(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    cbacks = []

    # if args.tboard:
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    cbacks.append(tensorboard)

    # if args.best:
    modfile = './model%d.h5' % int(time())
    mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
    cbacks.append(mcheck)

    history = model.fit(train_x, train_y, batch_size=config['training']['batch'],
                        epochs=config['training']['epochs'],
                        validation_data=(test_x, test_y),
                        verbose=1, callbacks=cbacks)
    # verbose=verbose -- he canviat a verbose=1

    ############################################
    # Store Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # poso timestamp als plots d'acc i loss
    import time
    ts = time.gmtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", ts)

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('music_loss_' + timestamp + '.pdf')
    plt.close()

    # Results

    # if args.best:
    model = load_model(modfile)

    score = model.evaluate(test_x, test_y, batch_size=config['training']['batch'], verbose=0)

    print()
    print('MSE test= ', score)
    print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
    test_yp = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)
    r2test = r2_score(test_y, test_yp)
    r2pers = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])   # no sé què fa!!!
    print('R2 test= ', r2test)
    print('R2 test persistence =', r2pers)

    #resfile = open('result-%s.txt' % config['data']['datanames'][0], 'a')
    resfile = open('results_%s.txt' % timestamp, 'w')
    resfile.write('LAG= %d, AHEAD= %d, RNN= %s, NLAY= %d, NNEUR= %d, DROP= %3.2f, ACT= %s, RACT= %s, '
                  'BATCH = %d, EPOCHS = %d, OPT= %s, LRATE = %3.5f, R2Test = %3.5f, R2pers = %3.5f\n' %
                  (config['data']['lag'],
                   config['data']['ahead'],
                   config['arch']['rnn'],
                   config['arch']['nlayers'],
                   config['arch']['neurons'],
                   config['arch']['drop'],
                   config['arch']['activation'],
                   config['arch']['activation_r'],
                   config['training']['batch'],
                   config['training']['epochs'],
                   config['training']['optimizer'],
                   config['training']['lrate'],
                   r2test, r2pers
                   ))
    resfile.close()

    # poso timestamp als arxius que creo per als valors de test_y i test_yp (predicted)
    test_y_file = open('test_y-%s.txt' % timestamp, 'w')
    test_y_file.write(np.array2string(test_y))
    test_y_file.close()
    test_yp_file = open('test_yp-%s.txt' % timestamp, 'w')
    test_yp_file.write(np.array2string(test_yp))
    test_yp_file.close()

    print('train_x: {}'.format(train_x))
    print('train_y: {}'.format(train_y))
    print('val_x: {}'.format(test_x))
    print('val_y: {}'.format(test_y))
    print('y_predicted: {}'.format(test_yp))

    # Deletes the model file
    try:
        os.remove(modfile)
    except OSError:
        pass
