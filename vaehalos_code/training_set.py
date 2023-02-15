from vaehalos_code import halo_data_processing as hdp
import pickle
import os


class TestSet:
    def __init__(self, params, sims_data, num_halos_per_sim=None, save=False, name="testset", **kwargs):
        path = params.saving_path + 'training_data/'
        scaler_output = open_pickle_file(path + 'scaler_output.pkl')
        scaler_queries = open_pickle_file(path + 'scaler_queries.pkl')
        try:
            ids, labels, queries = self.load_test_set(path, name)
            print("Loaded test set")
        except:
            ids, labels, queries = self.get_test_set(params, scaler_output, scaler_queries, save, path,
                                                     name, num_halos_per_sim, shuffle=False)
            print("Created a new test set")

        test_set = hdp.VAE_DataLoading_z0(ids, labels, queries, sims_data, params, shuffle=False, cache=False, **kwargs)
        self.testset_class = test_set
        self.test_set = test_set.get_dataset()
        self.scaler_queries = scaler_queries
        self.scaler_output = scaler_output

    @staticmethod
    def load_test_set(path, name):
        labels = open_pickle_file(path + '/' + str(name) + '_set_labels.pkl')
        ids = list(labels.keys())
        queries = open_pickle_file(path + '/' + str(name) + '_set_queries.pkl')
        return ids, labels, queries

    @staticmethod
    def get_test_set(params, scaler_output, scaler_queries, save=True, path=None, name=None, num_halos_per_sim=None,
                     shuffle=False):
        testset = hdp.VAE_InputsPreparation(params, params.val_sims, num_halos_per_sim=num_halos_per_sim,
                                            queries_idx=params.val_queries, shuffle=shuffle,
                                            scaler_output=scaler_output, scaler_queries=scaler_queries)
        ids = testset.halo_IDs
        labels = testset.labels_halo_IDS
        queries = testset.queries_halo_IDS

        if save:
            if path is None:
                path = params.saving_path + 'training_data'
            print("Saving test set and the parameters")
            save_pickle_file(path + '/' + str(name) + '_set_labels.pkl', labels)
            save_pickle_file(path + '/' + str(name) + '_set_queries.pkl', queries)
        return ids, labels, queries


class ValidationSet:
    def __init__(self, params, sims_data, training_set=None, num_halos_per_sim=None, save=False, path=None,
                 force_load=False, name="validation", shuffle=False, cache_path=None, force_new=False, **kwargs):
        if path is None:
            path = params.saving_path + 'training_data'
        if cache_path is None:
            cache_path = params.saving_path + 'training_data/vset'

        if force_load:
            ids, label, queries = self.load_validation_set(path, name)
            print("Loaded requested validation set at path " + str(path))
        elif force_new:
            print("Computing new validation set (forced to do so)")
            ids, label, queries = self.get_validation_set(params, training_set, save, path, name, num_halos_per_sim,
                                                           shuffle=shuffle)
        else:
            try:
                check_consistency_params(params, path)
                ids, label, queries = self.load_validation_set(path, name)
                print("Loaded existing validation set")

            except (FileNotFoundError, AssertionError, KeyError):
                print("Computing a new validation set")
                ids, label, queries = self.get_validation_set(params, training_set, save, path, name, num_halos_per_sim,
                                                              shuffle=shuffle)

        val_set = hdp.VAE_DataLoading_z0(ids, label, queries, sims_data, params, shuffle=shuffle,
                                         cache_path=cache_path, **kwargs)
        self.val_class = val_set
        self.val_set = val_set.get_dataset()

    @staticmethod
    def get_validation_set(params, training_set, save, path, name, num_halos_per_sim, shuffle=False):
        vset = hdp.VAE_InputsPreparation(params, params.val_sims, num_halos_per_sim=num_halos_per_sim,
                                         queries_idx=params.val_queries, shuffle=shuffle,
                                         scaler_output=training_set.scaler_output,
                                         scaler_queries=training_set.scaler_queries)
        ids = vset.halo_IDs
        labels = vset.labels_halo_IDS
        queries = vset.queries_halo_IDS

        if save:
            print("Saving validation set and the parameters")
            save_pickle_file(path + '/' + str(name) + '_set_labels.pkl', labels)
            save_pickle_file(path + '/' + str(name) + '_set_queries.pkl', queries)
        return ids, labels, queries

    @staticmethod
    def load_validation_set(path, name):
        labels = open_pickle_file(path + '/' + str(name) + '_set_labels.pkl')
        ids = list(labels.keys())
        queries = open_pickle_file(path + '/' + str(name) + '_set_queries.pkl')
        return ids, labels, queries


class TrainingSet:
    def __init__(self, params, sims_data, num_halos_per_sim=None, save=False, path=None, force_load=False, force_new=False,
                 shuffle=True, cache_path=None, **kwargs):
        if path is None:
            path = params.saving_path + 'training_data'
            if not os.path.exists(path):
                os.mkdir(path)
        if cache_path is None:
            cache_path = params.saving_path + 'training_data/tset'

        if force_load:
            ids, labels, queries, output_sclr, queries_sclr = self.load_training_set(path)
            print("Loaded requested training set at path " + str(path))

        elif force_new:
            print("Computing new training set (forced to do so)")
            ids, labels, queries, output_sclr, queries_sclr = self.get_training_set(params, save, path,
                                                                                    num_halos_per_sim, shuffle=shuffle)
        else:
            try:
                check_consistency_params(params, path)
                ids, labels, queries, output_sclr, queries_sclr = self.load_training_set(path)
                print("Loaded existing training set")

            except (FileNotFoundError, AssertionError, KeyError):
                print("Computing a new training set")
                ids, labels, queries, output_sclr, queries_sclr = self.get_training_set(params, save, path,
                                                                                        num_halos_per_sim, shuffle=shuffle)
                if save is True:
                    save_pickle_file(path + "/used_params.pkl", params)

        self.scaler_output = output_sclr
        self.scaler_queries = queries_sclr

        tr_class = hdp.VAE_DataLoading_z0(ids, labels, queries, sims_data, params, shuffle=shuffle,
                                          cache_path=cache_path, **kwargs)
        self.tr_class = tr_class
        self.training_set = tr_class.get_dataset()

    @staticmethod
    def get_training_set(params, save, path, num_halos_per_sim, shuffle=True):
        tset = hdp.VAE_InputsPreparation(params, params.tr_sims, num_halos_per_sim=num_halos_per_sim,
                                         queries_idx=params.tr_queries, shuffle=shuffle)
        ids = tset.halo_IDs
        labels = tset.labels_halo_IDS
        queries = tset.queries_halo_IDS

        out_scaler = tset.scaler_output
        query_scaler = tset.scaler_queries

        if save is True:
            print("Saving training set and the parameters")
            save_pickle_file(path + '/training_set_labels.pkl', tset.labels_halo_IDS)
            save_pickle_file(path + '/training_set_queries.pkl', tset.queries_halo_IDS)
            save_pickle_file(path + '/scaler_output.pkl', tset.scaler_output)
            save_pickle_file(path + '/scaler_queries.pkl', tset.scaler_queries)
        return ids, labels, queries, out_scaler, query_scaler

    @staticmethod
    def load_training_set(path):
        labels = open_pickle_file(path + '/training_set_labels.pkl')
        ids = list(labels.keys())
        queries = open_pickle_file(path + '/training_set_queries.pkl')
        out_scaler = open_pickle_file(path + '/scaler_output.pkl')
        query_scaler = open_pickle_file(path + '/scaler_queries.pkl')
        return ids, labels, queries, out_scaler, query_scaler


def open_pickle_file(filename):
    with open(filename, 'rb') as f:
        t = pickle.load(f)
    return t


def save_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def check_consistency_params(param1, path_used_params):
    # should check that params are the same as those used to compute training set first time
    used_params = open_pickle_file(path_used_params + "/used_params.pkl")
    compare_param_files(param1, used_params)


def compare_param_files(param1, param2):
    import copy
    _param2 = copy.deepcopy(dict(param2.__dict__))
    _param1 = copy.deepcopy(dict(param1.__dict__))
    del _param2['beta'], _param2['latent_dim'], _param1['beta'], _param1['latent_dim']
    assert all([x if isinstance(x, bool) else all(x) for key in _param2.keys()
                for x in [_param1[key] == _param2[key]]]), \
        "Used parameter and input parameters are not the same"

