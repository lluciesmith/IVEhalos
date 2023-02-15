import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.callbacks as callbacks
from dlhalos_code_tf2 import callbacks as mycallbacks
from dlhalos_code_tf2 import layers as mylayers


class MyVAEModel:
    def __init__(self, params, conv_params, fcc_params, scaler_output=None, beta_vary=False, num_gpu=1):
        self.params = params
        self.path = params.saving_path
        self.original_dim = params.dim
        self.latent_dim = params.latent_dim
        self.initialiser = params.initialiser
        self.conv_params = conv_params
        self.fcc_params = fcc_params
        self.beta_vary = beta_vary

        self.lr = params.lr
        # self.beta = params.beta
        # self.beta = mycallbacks.BetaCallback(params.beta).beta
        self.epochs = params.epochs
        self.early_stopping = params.early_stopping
        self.verbose = params.verbose
        self.tensorboard = params.tensorboard

        self.scaler_output = scaler_output

        # vae model
        if num_gpu > 1:
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            with strategy.scope():
                self.model()
        else:
            self.model()

    def model(self):
        self.convlayers = mylayers.ConvLayers(self.conv_params, self.initialiser)
        self.denselayers = mylayers.FCCLayers(self.fcc_params, self.initialiser)
        self.kl_layer = mylayers.KLLossLayer(self.params.beta)

        input_query = Input(shape=(1,), name="query")
        input_encoder = Input(shape=(*self.original_dim, 1), name='encoder_input')
        output_encoder = self.encoder(input_encoder)
        output_encoder = self.kl_layer(output_encoder)
        z_sample = self.sample(output_encoder)
        q_r = tf.keras.layers.Reshape((-1,))(input_query)
        input_decoder = tf.keras.layers.Concatenate(axis=1, name="input_decoder")([z_sample, q_r])
        output_decoder = self.decoder(input_decoder)
        vae = Model([input_encoder, input_query], output_decoder, name='vae')
        self.vae = vae
        self.encoder_model = Model(input_encoder, output_encoder, name='encoder')
        self.decoder_model = self.get_decoder_model()

    def encoder(self, inputs):
        x = self.convlayers.conv_layers(inputs, self.original_dim)
        x = tf.keras.layers.Flatten()(x)
        bottleneck = Dense(self.latent_dim * 2)(x)
        z_mean, z_log_var = tf.split(bottleneck, 2, axis=1, name="latent")
        return z_mean, z_log_var

    def sample(self, args):
        z_mean, z_log_var = args
        # Use reparametrization trick to sample from gaussian
        # tf.random.set_seed(5)
        epsilon = tf.random.normal(shape=tf.shape(z_mean), seed=0)
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

    def decoder(self, latent_inputs):
        y = self.denselayers.fcc_layers(latent_inputs)
        outputs = Dense(1, **self.fcc_params['last'], kernel_initializer=self.initialiser, name='prediction_layer')(y)
        return outputs

    def compile_model(self):
        optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        # loss = losses.cauchy_selection_loss_fixed_boundary(gamma=0.2)
        loss = 'mse'
        self.vae.compile(optimizer=optimiser, loss=loss, metrics=[loss])

    def get_decoder_model(self):
        names = [layer.name for layer in self.vae.layers]
        idx = names.index('input_decoder') + 1
        shape = self.vae.layers[idx].get_input_shape_at(0)
        input_dec = Input(shape=shape)
        x = input_dec
        idx_final = names.index('prediction_layer')
        for layer in self.vae.layers[idx: idx_final+1]:
            x = layer(x)
        return Model(input_dec, x, name='decoder')

    # training

    def train(self, training_set, validation_set, callbacks=None, initial_epoch=0):
        if initial_epoch == 0:
            # compile the model only if you are starting the training from scratch
            self.compile_model()
        if callbacks is None:
            callbacks = self.get_callbacks()
        history = self.vae.fit(training_set, epochs=self.epochs, validation_data=validation_set,
                               verbose=self.verbose, callbacks=callbacks, initial_epoch=initial_epoch)
        self.history = history
        self.vae.save(self.path + "vae_model.h5")

    def get_callbacks(self):
        callbacks_list = []

        # checkpoint
        if not os.path.exists(self.path + "model"):
            os.mkdir(self.path + "model")
        checkpoint_call = callbacks.ModelCheckpoint(self.path + "model/model.{epoch:02d}.h5", save_freq='epoch',
                                                    save_weights_only=False)
        callbacks_list.append(checkpoint_call)

        # beta-varying with epoch
        if self.beta_vary:
            bcall = mycallbacks.BetaCallback(self.vae.get_layer("KLlayer").non_trainable_weights[0], self.params)
            callbacks_list.append(bcall)

        # early stopping
        if self.early_stopping:
            estop = callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=self.verbose,
                                            restore_best_weights=True)
            callbacks_list.append(estop)

        # Record training history in log file
        csv_logger = callbacks.CSVLogger(self.path + "training.log", separator=',', append=True)
        callbacks_list.append(csv_logger)

        # tensorboard
        if self.tensorboard:
            c = tf.keras.callbacks.TensorBoard(log_dir=self.path + 'logs', histogram_freq=1, write_graph=False,
                                               write_images=False, update_freq='epoch', profile_batch=(10, 20),
                                               embeddings_freq=0, embeddings_metadata=None)
            callbacks_list.append(c)

        return callbacks_list

    # load weights
    def load(self, path=None, num_epoch=None):
        if path is None:
            path = self.path + "model/model.%02d.h5" % num_epoch
        print("Loading model from " + path)
        self.vae = keras.models.load_model(path, custom_objects={"KLLossLayer": mylayers.KLLossLayer})

        num_layers = len(self.encoder_model.get_weights())
        w_vae = self.vae.get_weights()
        self.encoder_model.set_weights(w_vae[:num_layers])
        self.decoder_model.set_weights(w_vae[num_layers:])

    def load_weights(self, path_weights=None, num_epoch=None):
        if path_weights is None:
            path_weights = self.path + "model/weights.%02d.h5" % num_epoch
        print("Loading weights from " + path_weights)
        self.vae.load_weights(path_weights)

    # predict

    def predict_latent_mean_std(self, dataset, sim_id='11', epoch=None, save=False):
        z_mean, z_var = self.encoder_model.predict(dataset, verbose=self.verbose)
        z_std = np.exp(0.5 * z_var)

        if save:
            if epoch is None:
                epoch = self.epochs
            np.save(self.path + "latent_rep_" + sim_id + "_epoch_%02d.npy" % epoch, [z_mean, z_std])
        return z_mean, z_std

    def predict(self, dataset, scaler=None, sim_id='11', epoch=None, save=False, sampling=True):
        if epoch is None:
            epoch = self.epochs

        if sampling is False:
            results = self._predict_no_sampling(dataset, scaler=scaler)
            filename = self.path + "truths_pred_sim_" + sim_id + "_epoch_%02d_no_sampling.npy" % epoch
        else:
            results = self._predict_with_sampling(dataset, scaler=scaler)
            filename = self.path + "truths_pred_sim_" + sim_id + "_epoch_%02d.npy" % epoch

        if save:
            np.save(filename, results)
            print("Saved predictions at path " + self.path)
        return results

    def predict_n_queries(self, dataset_class, n_queries=100, sim_id='11', epoch=None, save=False):
        dc = dataset_class.testset_class

        haloids = np.array([int(ID[ID.find('-id-') + len('-id-'): ID.find('-query-')]) for ID in dc.list_IDs])
        hids_unique = np.unique(haloids)

        queries = np.array([int(ID[ID.find('-query-') + len('-query-'):]) for ID in dc.list_IDs])
        qlims = [(min(x), max(x)) for hID in hids_unique
                 for x in [[dc.queries['sim-11-id-%i-query-%i' % (hID, q)] for q in queries[haloids == hID]]]]
        qs = np.array([np.linspace(qlim[0], qlim[1], n_queries) for qlim in qlims])

        rhos_n = np.zeros((len(hids_unique), n_queries))
        for n in range(len(hids_unique)):
            inputs = dc.get_input('sim-11-id-%i-query-%i' % (hids_unique[n], min(queries)))
            output_encoder = self.encoder_model(inputs.reshape(1, *inputs.shape))
            zsample = self.sample(output_encoder)
            input_dec = np.column_stack((np.repeat(zsample, len(qs[n]), axis=0), qs[n]))
            pred = self.decoder_model(tf.constant(input_dec, dtype=tf.float32)).numpy()
            rhos_n[n] = dataset_class.scaler_output.inverse_transform(pred).flatten()

        r = np.column_stack((hids_unique, dataset_class.scaler_queries.inverse_transform(qs.reshape(-1, 1)).flatten()))
        rho = np.column_stack((hids_unique, rhos_n))
        if save:
            np.save(self.path + "many_queries_truths_pred_sim_" + sim_id + "_epoch_%02d.npy" % epoch, rho)
            np.save(self.path + "many_queries_" + sim_id + "_epoch_%02d.npy" % epoch, r)
        return r, rho

    def _predict_with_sampling(self, dataset, scaler=None):
        pred = self.vae.predict(dataset, verbose=self.verbose)

        if scaler is None:
            scaler = self.scaler_output
        results = self._transform_predictions(pred, dataset, scaler)
        return results

    def _predict_no_sampling(self, dataset, scaler=None):
        z_mean, z_var = self.encoder_model.predict(dataset, verbose=self.verbose)
        input_decoder = tf.keras.layers.Concatenate(axis=1)([z_mean, dataset['query']])
        pred = self.decoder_model.predict(input_decoder, verbose=self.verbose)

        if scaler is None:
            scaler = self.scaler_output
        results = self._transform_predictions(pred, dataset, scaler)
        return results

    def _p_no_sampling2(self, dataset, scaler=None):
        p = []
        for elem in dataset:
            test_sample, query = elem[0]['encoder_input'], elem[0]['query']
            mean, logvar = self.encoder_model(test_sample)
            input_decoder = tf.keras.layers.Concatenate(axis=1)([mean, query])
            predictions = self.decoder_model(input_decoder)
            p.append(predictions)

        p = np.concatenate(p, axis=0)
        if scaler is None:
            scaler = self.scaler_output
        results = self._transform_predictions(p, dataset, scaler)
        return results

    @staticmethod
    def _transform_predictions(predictions, dataset, scaler):
        _pred = np.concatenate(predictions)
        pred = scaler.inverse_transform(_pred.reshape(-1, 1)).flatten()

        _truths = np.concatenate([elem[1].numpy() for elem in dataset])
        truths = scaler.inverse_transform(_truths.reshape(-1, 1)).flatten()

        result = np.column_stack((truths, pred))
        return result







