import numpy as np
import h5py
from collections import OrderedDict
import pynbody
from dlhalos_code_tf2 import rescaling
from tensorflow.keras.utils import Sequence
import tensorflow as tf


class VAE_InputsPreparation:
    def __init__(self, params, sims, num_halos_per_sim=None, queries_idx=None, scaler_output=None, scaler_queries=None, shuffle=None):
        """
        This class prepares the inputs in the correct format for the DataGenerator class.
        It generates two dictionaries, one for labels and one for queries, of format {halo: label} or {halo: query},
        where `halo` is a string 'sim-#-id-#-query-#' % (sim ID, halo ID, query number).

        This class loads a file named ``reseed#_halo_data.txt'' where # is the sim ID, s.t.
        column 0 = halo ID, first N columns = queries, second N columns = labels, last column = r200 [kpc/h],
        and N is the number of queries defined by `params.num_queries`.

        """
        self.params = params
        self.sims = sims
        self.path = params.path_sims

        self.num_queries = params.tot_num_queries
        self.num_halos = num_halos_per_sim
        self.queries_idx = queries_idx
        if self.queries_idx is None:
            self.queries_idx = np.arange(self.num_queries)

        self.shuffle = shuffle
        if shuffle is None:
            self.shuffle = params.shuffle

        # Rescaling query and label variables
        self.scaler_queries = scaler_queries
        self.scaler_output = scaler_output

        self.halo_IDs = None
        self.labels_halo_IDS = None
        self.queries_halo_IDS = None
        self.generate_halo_IDs_dictionary()

    def generate_halo_IDs_dictionary(self):
        flat_name, flat_query, flat_label = self.get_ids_query_mass()

        # Rescale the labels
        if self.params.return_rescaled_labels is True:
            flat_label, self.scaler_output = rescaling.rescale_output(flat_label, self.params.scaler_type,
                                                                      self.params.output_range, self.scaler_output)
        # Rescale the queries
        if self.params.return_rescaled_queries is True:
            flat_query, self.scaler_queries = rescaling.rescale_output(flat_query, self.params.scaler_type_queries,
                                                                       self.params.queries_range, self.scaler_queries)

        # Construct dictionary
        dict_i_label = OrderedDict(zip(flat_name, flat_label))
        dict_i_query = OrderedDict(zip(flat_name, flat_query))

        # Shuffle once when assigning ID to training, validation, and test sets.
        if self.shuffle is True:
            dict_i_label, dict_i_query = self.shuffle_dictionary([dict_i_label, dict_i_query])

        self.halo_IDs = list(dict_i_label.keys())
        self.labels_halo_IDS = dict_i_label
        self.queries_halo_IDS = dict_i_query

    # Functions to generate list of haloID, their labels and their queries

    def get_ids_query_mass(self):
        names, queries, labels = [], [], []

        for i, sim_ID in enumerate(self.sims):
            ids_names, query_i, label_i = self.extract_halos_and_properties(sim_ID)
            names.append(ids_names)
            queries.append(query_i)
            labels.append(label_i)

        return np.concatenate(names), np.concatenate(queries), np.concatenate(labels)

    def load_halo_data(self, sim_ID):
        num_queries = self.num_queries
        path = self.path + "L50_N512_" + str(sim_ID)
        hdata_ = np.loadtxt(path + "/reseed" + str(sim_ID) + "_" + self.params.halodatafile + ".txt", delimiter=",")

        # Remove halos which have inf values in their profiles
        rm_halos = np.where(np.any(np.isinf(hdata_[:, 1 + num_queries:1 + 2*num_queries]), axis=1))[0]
        hdata = np.delete(hdata_, tuple(rm_halos), axis=0)
        assert len(np.where(np.any(np.isinf(hdata[:, 1 + num_queries:1 + 2*num_queries]), axis=1))[0]) == 0
        return hdata

    def extract_halos_and_properties(self, sim_ID):
        num_queries = self.num_queries
        halodata = self.load_halo_data(sim_ID)

        cols_queries = np.arange(1, 1 + num_queries)[self.queries_idx]
        cols_labels = np.arange(1 + num_queries, 1 + 2*num_queries)[self.queries_idx]

        rows = np.where(halodata[:, -1] <= self.params.R200max)[0] if self.params.R200max is not False else np.arange(len(halodata))
        if self.num_halos is not None:
            rows = np.random.choice(rows, self.num_halos, replace=False)

        hids = self.get_name_tag_samples(sim_ID, halodata[rows, 0].astype("int"), self.queries_idx)
        queries = halodata[rows][:, cols_queries].flatten('C')
        labels = halodata[rows][:, cols_labels].flatten('C')
        return hids, queries, labels

    @staticmethod
    def get_name_tag_samples(sim_ID, halo_ids, queries_indices):
        name = []
        for id_i in halo_ids:
            for bin in queries_indices:
                name.append('sim-' + str(sim_ID) + '-id-' + str(id_i) + '-query-' + str(bin))
        return name

    @staticmethod
    def shuffle_dictionary(diction):
        if isinstance(diction, list):
            assert [set(diction[0].keys()) == set(diction[i].keys()) for i in range(len(diction))]
            ids_reordering = np.random.permutation(list(diction[0].keys()))
            dict_shuffled = [OrderedDict([(key, d[key]) for key in ids_reordering]) for d in diction]

        else:
            ids_reordering = np.random.permutation(list(diction.keys()))
            dict_shuffled = OrderedDict([(key, diction[key]) for key in ids_reordering])
        return dict_shuffled


class VAE_DataGenerator_z0(Sequence):
    def __init__(self, list_IDs, labels, queries, sims, params, shuffle=False, num_threads=None):
        """
        This class created the data generator that should be used to fit the deep learning model.
        :param list_IDs: String of form 'sim-%i-id-%i-query-%i' % (simulation_index, halo_ID, query_number)
        :param labels: This is a dictionary of the form {halo ID: labels}
        :param queries: This is a dictionary of the form {halo ID: queries}
        :param sims: list of simulation IDs
        :param params: Parameters for width/resolution of sub-box + res/path of simulations
        :param shuffle: Shuffle the tf.data dataset
        """

        self.list_IDs = list_IDs
        self.num_IDs = len(list_IDs)
        self.labels = labels    # dictionary of the type {halo-ID name: label}
        self.queries = queries  # dictionary of the type {halo-ID name: query}

        self.sims = sims
        self.path = params.path_sims

        self.shuffle = shuffle
        self.dim = params.dim
        self.res = self.dim[0]
        self.width = params.width

        self.batch_size = params.batch_size
        self.n_channels = params.n_channels

        self.num_threads = num_threads
        if num_threads is None:
            self.num_threads = tf.data.experimental.AUTOTUNE

        if params.dtype == "float64":
            self.Tout_dtype = tf.float64
        else:
            self.Tout_dtype = tf.float32

        self.rho_m = {}
        self.halos_pos = {}
        for i, sim in self.sims.items():
            self.rho_m[i] = sim.properties['rhoM']
            self.halos_pos[i] = self.get_halo_positions(sim)

    def data_generation(self, idx):
        idx = int(idx)
        ID = self.list_IDs[idx]
        sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id-')]
        halo_ID = int(ID[ID.find('-id-') + len('-id-'): ID.find('-query-')])

        s = self.get_subbox(sim_index, halo_ID)
        box = s.reshape((*self.dim, self.n_channels))
        query = self.queries[ID]
        boxlabel = self.labels[ID]
        return box, query, boxlabel

    def tf_data_gen(self, idx):
        b, q, l = tf.py_function(func=self.data_generation, inp=[idx],
                                 Tout=(self.Tout_dtype, self.Tout_dtype, self.Tout_dtype))
        out = {"encoder_input": b, "query": q}, l
        return out

    def get_dataset(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(tf.range(self.num_IDs))
        if self.shuffle is True:
            dataset = dataset.shuffle(self.num_IDs)
        dataset = dataset.map(self.tf_data_gen, num_parallel_calls=self.num_threads)
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    # Generate input from scratch

    def get_subbox(self, sim_index, haloID):
        sim = self.sims[sim_index]
        width = self.width
        resolution = self.res

        x, y, z = self.halos_pos[sim_index][haloID]
        edges = (x-(width/2), y-(width/2), z-(width/2), x+(width/2), y+(width/2), z+(width/2))
        subsnap = sim[pynbody.filt.Cuboid(*edges)]
        rho_grid = pynbody.sph.to_3d_grid(subsnap, qty="rho", nx=resolution, threaded=True)
        return np.log10(rho_grid/self.rho_m[sim_index] + 1)

    @staticmethod
    def get_halo_positions(sim_snapshot):
        h = sim_snapshot.halos(grp_array=True)
        with h5py.File(h.halofilename, 'r') as f:
            pos = f['Group']['GroupPos'][:]
            pos = pynbody.array.SimArray(pos, 'Mpc a h**-1')
            pos.sim = sim_snapshot
        pos.convert_units(sim_snapshot['pos'].units)
        return pos


class VAE_DataLoading_z0(Sequence):
    def __init__(self, list_IDs, labels, queries, sims, params,
                 shuffle=False, cache=True, prefetch=True, num_threads=None, cache_path=None, drop_remainder=False):
        """
        This class created the data generator that should be used to fit the deep learning model.
        :param list_IDs: String of form 'sim-%i-id-%i-query-%i' % (simulation_index, halo_ID, query_number)
        :param labels: This is a dictionary of the form {halo ID: labels}
        :param queries: This is a dictionary of the form {halo ID: queries}
        :param sims: list of simulation IDs
        :param params: Parameters for width/resolution of sub-box + res/path of simulations
        :param shuffle: Shuffle the tf.data dataset
        """

        self.list_IDs = list_IDs
        self.num_IDs = len(list_IDs)
        self.labels = labels    # dictionary of the type {halo-ID name: label}
        self.queries = queries  # dictionary of the type {halo-ID name: query}
        assert self.list_IDs == list(self.labels.keys()) == list(self.queries.keys())

        self.sims = sims
        self.path = params.path_sims
        self.cache_path = cache_path
        if cache_path is None:
            self.cache_path = self.path

        self.shuffle = shuffle
        self.cache = cache
        self.prefetch = prefetch
        self.drop_remainder = drop_remainder

        self.dim = params.dim
        self.res = self.dim[0]
        self.width = params.width
        self.num_threads = num_threads
        if num_threads is None:
            self.num_threads = tf.data.experimental.AUTOTUNE

        self.batch_size = params.batch_size
        self.n_channels = params.n_channels

        if params.dtype == "float64":
            self.Tout_dtype = tf.float64
        else:
            self.Tout_dtype = tf.float32

    def get_dataset(self):
        output_signature = tf.TensorSpec(shape=(), dtype=tf.string), \
                           tf.TensorSpec(shape=(), dtype=tf.float32),\
                           tf.TensorSpec(shape=(), dtype=tf.float32)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        if self.shuffle is True:
            dataset = dataset.shuffle(self.num_IDs)
        dataset = dataset.map(self.map_generator, num_parallel_calls=self.num_threads)
        if self.cache is True:
            dataset = dataset.cache(self.cache_path)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        if self.prefetch is True:
            dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def generator(self):
        for sample in self.list_IDs:
            yield sample, self.queries[sample], self.labels[sample]

    def map_generator(self, x_elem, query_elem, label_elem):
        x_input = tf.numpy_function(func=self.get_input, inp=[x_elem], Tout=self.Tout_dtype)
        return {"encoder_input": x_input, "query": query_elem}, label_elem

    def get_input(self, ID):
        ID = str(ID)
        sim_ID = ID[ID.find('sim-') + len('sim-'): ID.find('-id-')]
        h_ID = int(ID[ID.find('-id-') + len('-id-'): ID.find('-query-')])

        path = self.path + "L50_N512_" + sim_ID + "/reseed" + sim_ID
        with h5py.File(path + "_halo_inputs_width_%.1f_res_%i.hdf5" % (self.width, self.res), 'r') as f:
            inputs_file = f['inputs'][int(np.where(f['haloIDs'][:] == h_ID)[0])]

        return inputs_file.reshape((*self.dim, self.n_channels))

    # # yield box input directly from generator
    #
    # def get_dataset2(self):
    #     output_signature = (tf.TensorSpec(shape=(131, 131, 131, 1), dtype=tf.float32, name="encoder_input"),
    #                         tf.TensorSpec(shape=(), dtype=tf.float32, name="query")),\
    #                        tf.TensorSpec(shape=(), dtype=tf.float32)
    #     AUTOTUNE = tf.data.experimental.AUTOTUNE
    #     dataset = tf.data.Dataset.from_generator(self.generator2, output_signature=output_signature)
    #     if self.shuffle is True:
    #         dataset = dataset.shuffle(100)
    #     if self.cache is True:
    #         dataset = dataset.cache()
    #     dataset = dataset.batch(self.batch_size)
    #     if self.prefetch is True:
    #         dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    #     return dataset
    #
    # def generator2(self):
    #     filename = self.path + 'halos_inputs_datawidth_%.1f_res_%i.hdf5' % (self.width, self.res)
    #     with h5py.File(filename, 'r') as f:
    #         for sample in self.list_IDs:
    #             box = f[sample[:sample.find('-query-')]][:].reshape((*self.dim, self.n_channels))
    #             yield (box, self.queries[sample]), self.labels[sample]


