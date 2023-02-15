from vaehalos_code import params
import numpy as np
import os
import h5py
import pandas as pd


def resume_training_script(path_params, resume_epoch, num_gpu=1):
    from vaehalos_code import svae2
    from vaehalos_code import training_set as trdata
    from dlhalos_code_tf2 import z0_data_processing as z0dp

    k, params_model = load_params(path_params)

    # Prepare the simulations you will use for training/validation
    s = z0dp.SimulationPreparation_z0(k, path="/freya/ptmp/mpa/luisals/Simulations/")

    # Get training set and validation set
    tset = trdata.TrainingSet(k, s.sims_dic, save=True, force_load=True)
    vset = trdata.ValidationSet(k, s.sims_dic, training_set=tset, num_halos_per_sim=2000, save=True, force_load=True)
    print("Loading path file for training data is: ")
    print(k.saving_path)

    k.saving_path = k.saving_path + "/beta" + str(k.beta) + "/latent" + str(k.latent_dim) + "/"
    print("Saving path file for model is: ")
    print(k.saving_path)
    M = svae2.MyVAEModel(k, params_model.param_conv, params_model.param_fcc, scaler_output=tset.scaler_output,
                         beta_vary=True, num_gpu=num_gpu)
    M.load(num_epoch=resume_epoch)
    M.train(tset.training_set, vset.val_set, initial_epoch=resume_epoch)


def training_script(path_params, num_gpu=1):
    from vaehalos_code import svae2
    from vaehalos_code import training_set as trdata
    from dlhalos_code_tf2 import z0_data_processing as z0dp

    k, params_model = load_params(path_params)

    # Prepare the simulations you will use for training/validation
    s = z0dp.SimulationPreparation_z0(k, path="/freya/ptmp/mpa/luisals/Simulations/")

    # Get training set and validation set
    tset = trdata.TrainingSet(k, s.sims_dic, save=True, force_load=k.force_load, drop_remainder=True)
    vset = trdata.ValidationSet(k, s.sims_dic, training_set=tset, num_halos_per_sim=2000, save=True,
                                force_load=k.force_load, drop_remainder=True)
    print("Saving/loading path file for training data is: ")
    print(k.saving_path)

    k.saving_path = k.saving_path + "/beta" + str(k.beta) + "/latent" + str(k.latent_dim) + "/"
    if not os.path.exists(k.saving_path):
        os.makedirs(k.saving_path)
    print("Saving path file for model is: ")
    print(k.saving_path)
    M = svae2.MyVAEModel(k, params_model.param_conv, params_model.param_fcc, scaler_output=tset.scaler_output,
                         beta_vary=True, num_gpu=num_gpu)
    M.train(tset.training_set, vset.val_set)


def predictions_script(path_params, epoch=None, mutual_info=True, bws=None):
    from vaehalos_code import svae2
    from vaehalos_code import training_set as trdata
    from vaehalos_code import mutual_info_utils as miu
    from dlhalos_code_tf2 import z0_data_processing as z0dp

    # Load the paramter files
    k, params_model = load_params(path_params)

    if epoch is None:
        tr = pd.read_csv(k.saving_path + "/beta" + str(k.beta) + "/latent" + str(k.latent_dim) + '/training.log', sep=",", header=0)
        epoch = [np.argmin(tr['val_mse']) + 1, np.argmin(tr['val_KL']) + 1]

    if isinstance(epoch, (int, np.int64, np.int32)):
        epoch = [epoch]
    if bws is None:
        bws = [0.1, 0.2, 0.3]

    # Prepare the simulations you will use for training/validation
    s = z0dp.SimulationPreparation_z0(k, path="/freya/ptmp/mpa/luisals/Simulations/")

    # Get training set and validation set
    test_set = trdata.TestSet(k, s.sims_dic, save=True)
    print("Saving test-set path file is: ")
    print(k.saving_path)

    # Load the model
    k.saving_path = k.saving_path + "/beta" + str(k.beta) + "/latent" + str(k.latent_dim) + "/"
    for num_epoch in epoch:
        print("Making predictions for epoch " + str(num_epoch))
        M = svae2.MyVAEModel(k, params_model.param_conv, params_model.param_fcc, scaler_output=test_set.scaler_output,
                             beta_vary=True)
        M.load(num_epoch=num_epoch)
        p = M.predict(test_set.test_set, sim_id=k.val_sims[0] + "_testset_", save=True, epoch=num_epoch)
        lmean, lstd = M.predict_latent_mean_std(test_set.test_set, sim_id=k.val_sims[0] + "_testset_", save=True, epoch=num_epoch)
        rhotrue, lmean, lstd = rearrange_results(k, test_set, p, lmean, lstd, num_epoch, save=True)

        if mutual_info:
            mi_values = []
            for bw in bws:
                mi_kl_truth = miu.KL_mi_truth_latents(rhotrue, lmean, lstd, nsamples=1, bandwidth=bw, pool=True)
                mi_values.append(mi_kl_truth)
                np.save(k.saving_path + "mi_truth_latents_epoch_%i_bw_%.1f.npy" % (num_epoch, bw), mi_kl_truth)
            add_mi_values_to_hdf5_file(k, num_epoch, mi_values, bws)

        del M, p, lmean, lstd


def rearrange_results(paramfile, testset, predictions, latent_mean, latent_std, num_epoch, save=True):
    # Re-arrange predictions
    num_rbins = len(paramfile.val_queries)
    testids = np.array([int(x[x.find('-id-') + len('-id-'): x.find('-query-')]) for x in testset.testset_class.list_IDs])
    testids = testids[::num_rbins]
    radii = np.array([testset.testset_class.queries[x] for x in testset.testset_class.list_IDs])
    radii = testset.scaler_queries.inverse_transform(radii.reshape(-1, 1)).reshape(len(testids), num_rbins)
    rho_true = np.array([testset.testset_class.labels[x] for x in testset.testset_class.list_IDs])
    rho_true = testset.scaler_output.inverse_transform(rho_true.reshape(-1, 1)).reshape(len(testids), num_rbins)
    np.allclose(predictions[:, 0].reshape(len(testids), num_rbins), rho_true)
    rho_pred = predictions[:, 1].reshape(len(testids), num_rbins)
    lmean = latent_mean[::num_rbins, :]
    lstd = latent_std[::num_rbins, :]
    if save:
        filename = paramfile.saving_path + "results_testset_" + paramfile.val_sims[0] + "_epoch_%i.hdf5" % num_epoch
        testset_results = h5py.File(filename, 'w')
        testset_results.create_dataset('haloIDs', data=testids, maxshape=(len(testids),))
        testset_results.create_dataset('rho_true', data=rho_true, maxshape=(len(testids), num_rbins))
        testset_results.create_dataset('rho_pred', data=rho_pred, maxshape=(len(testids), num_rbins))
        testset_results.create_dataset('r', data=radii, maxshape=(len(testids), num_rbins))
        testset_results.create_dataset('lmean', data=lmean, maxshape=(len(testids), paramfile.latent_dim))
        testset_results.create_dataset('lstd', data=lstd, maxshape=(len(testids), paramfile.latent_dim))
        testset_results.close()
    return rho_true, lmean, lstd


def add_mi_values_to_hdf5_file(paramfile, num_epoch, mi_values, bws):
    filename = paramfile.saving_path + "results_testset_" + paramfile.val_sims[0] + "_epoch_%i.hdf5" % num_epoch
    results = h5py.File(filename, 'r+')
    if mi_values is not None:
        for j, bw in enumerate(bws):
            results.create_dataset('MI_latents_truth_mean_bw%.1f' % bw, data=mi_values[j])
    results.close()


def load_params(path_params):
    k = params.Params(path_params + "/param_file.txt")
    params_model = params.Architecture(path_params + "/param_model.txt")
    print("Parameter file path is: ")
    print(path_params)
    print("Latent dim is: ")
    print(k.latent_dim)
    print("Beta is " + str(k.beta))
    return k, params_model

