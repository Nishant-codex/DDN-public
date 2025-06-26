import numpy as np
import pickle as pkl
from reservoirpy import datasets
from simulator import NetworkSimulator
from utils import single_sample_NRSE, eval_candidate_signal_gen_horizon
from datetime import date
import os 
import argparse
from tqdm import tqdm
from utils import createNARMA30
from utils import eval_candidate_lag_gridsearch_NARMA
# narma_INPUT = createNARMA30(200)

def resample_net_NARMA_best(data_dict, maxgen=None):
    validation_scores = data_dict['validation performance']
    if not maxgen is None:
        validation_scores = validation_scores[:maxgen, :, :]
    score = np.mean(validation_scores, axis=-1)
    max_gens = np.max(score, axis=-1)
    best_gen = np.argmax(max_gens)
    best_pop = score[best_gen, :]
    best_ind = np.argmax(best_pop)
    all_params = data_dict['parameters']
    best_params = all_params[best_gen, best_ind]
    net = data_dict['example net']
    best_net = net.get_new_network_from_serialized(best_params)
    return best_net

def resample_net_MG_worst(data_dict):
    validation_scores = data_dict['validation performance']
    score = np.min(validation_scores, axis=-1)
    max_gens = np.max(score, axis=-1)
    best_gen = np.argmax(max_gens)
    best_pop = score[best_gen, :]
    best_ind = np.argmax(best_pop)
    all_params = data_dict['parameters']
    # best_params = all_params[100, 8]
    best_params = all_params[best_gen, best_ind]
    net = data_dict['example net']
    best_net = net.get_new_network_from_serialized(best_params)
    return best_net

def retrain_net_NARMA(best_net, data_dict,):
    best_net.reset_network()
   
    
    val_performance_per_lag, model_per_lag = eval_candidate_lag_gridsearch_NARMA(best_net, data_dict['train data'],data_dict['validation data'], warmup=400,
                                        lag_grid=range(0, 15), alphas=[10e-14, 10e-13, 10e-12])
    return val_performance_per_lag, model_per_lag,  best_net

def test_net_NARMA(network, model, error_margin, test_data):
    warmup = 400
    model = model[0]  # get the first model, since we only use one lag
    prediction_steps_across_sequences = []
    y_across_sequences = []
    max_it_val = 500
    sim = NetworkSimulator(network)

    for sequence in test_data:
        start_input_val = sequence[0][warmup]
        labels_val = sequence[1][warmup + 1:]
        sim.warmup(sequence[0][:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val
        label_variance = np.var(labels_val)
        steps = 0
        y = []
        while j <= max_it_val:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            if len(output.shape) == 1:
                output = np.expand_dims(output, 0)
            feedback_in = model.predict(output)[0]
            y.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            j += 1
            if error <= error_margin:
                steps += 1

        prediction_steps_across_sequences.append(steps)
        y_across_sequences.append(y)
    return y_across_sequences, prediction_steps_across_sequences

def get_validation_throughout_evolution(dict, gen_max, populations=False, multitask=False):
    """_summary_

    Args:
        dict (_type_): _description_
        gen_max (_type_): _description_
        populations (bool, optional): _description_. Defaults to False.
        multitask (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Validation scores are saved in a matrix with dimensions:
    # generations x hyperparameter candidates x re-initializations x lag search grid
    if not multitask:
        all_scores = dict['validation performance'][:gen_max, :, :, :]
    else:
        all_scores = dict['validation performance'][:gen_max, :, :, :, :]
    # from the lag search grid we select the best score (lowest NRMSE), since this was the best 
    # performing readout model
    best_lag_scores = np.min(all_scores, axis=-1)
    
    if multitask:
        # from the tasks, we select the average
        # best_lag_scores = best_lag_scores[:, :, :, 0]**2
        best_lag_scores = np.mean(best_lag_scores, axis=-1)
    
    # from the re-initializations from the same hyperparameter set/candidate, we take the average
    best_candidate_scores = np.mean(best_lag_scores, axis=-1)
    
    if populations:
        return best_candidate_scores
    
    # from the population of hyperparameter candidates, we select the best candidate
    best_gen_scores = np.min(best_candidate_scores, axis=-1)
    return best_gen_scores

def get_best_candidate(dict, gen_max, multitask=False):
    """_summary_

    Args:
        dict (_type_): _description_
        gen_max (_type_): _description_
        multitask (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    best_gen_scores = get_validation_throughout_evolution(dict, gen_max, multitask=multitask)   
    best_candidates = get_validation_throughout_evolution(dict, gen_max, populations=True, multitask=multitask)
    best_gen_ind = np.argmin(best_gen_scores)
    best_candidate_ind = np.argmin(best_candidates[best_gen_ind])
    return best_gen_ind, best_candidate_ind
    
def sample_best_net(dict, n, gen_max, multitask=False):
    """_summary_

    Args:
        dict (_type_): _description_
        n (_type_): _description_
        gen_max (_type_): _description_
        multitask (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    gen, ind = get_best_candidate(dict, gen_max, multitask=multitask)
    # print(gen, ind)
    start_net = dict['example net']
    best_parameter_set = dict['parameters'][gen, ind, :]
    best_nets = []
    for i in range(n):
        best_net = start_net.get_new_network_from_serialized(best_parameter_set)
        best_nets.append(best_net)
    return best_nets

def testVisualize(network, data):
    sim = NetworkSimulator(network, False)
    sim.visualize(data)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Experiment configuration",
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("path", action="store", type=str, default="./", help="Evolution results data path")
    # parser.add_argument("-r", "--resamples", action="store", type=int, default=100, help="Number of resamples per test")
    # parser.add_argument("-t", "--testsamples", action="store", type=int, default=502, help="Test sequence length")
    # parser.add_argument("-s", "--testsequences", action="store", type=int, default=5, help="Number of test sequences per network")
    # parser.add_argument("-g", "--maxgen", action="store", type=int, default=None, help="Takes best up to this generation")


    # args = parser.parse_args()
    # config = vars(args)
    resamples = 100 #config['resamples']
    n_test_samples = 502 #config['testsamples']
    n_test_sequences = 5 #config['testsequences']
    path_dir = "./heterogeneity_results/" #config['path']
    maxgen = 150 #config['maxgen']
    save_dir = "./test_results_NARMA/"
    # Load data
    for path in os.listdir(path_dir):
    # path  = "2024-09-26_single_task_exp_BL_dist_decay_net_wide_gen150_test_optimized.p"
        print("Loading hyperparameter optimization results from " + path)

        with open(path_dir+path, 'rb') as f:
            results_dict = pkl.load(f)


        # Generate test data
        n_test_samples = 502
        warmup = 400

        test_data = []
        for seq in range(n_test_sequences):
            test_sequence = createNARMA30(n_test_samples + warmup)
            test_data.append(test_sequence)

        test_results = []

        resampled_networks = []
        print("Sample networks")

        for resample in range(resamples):
            # best_net = resample_net_MG_worst(results_dict)
            # testVisualize(best_net, test_data_tau[14][0])
            best_net = sample_best_net(results_dict,n=1, gen_max=maxgen)
            resampled_networks.append(best_net)

        error_margin = 0.2 #results_dict['error margin']
        progress_bar = tqdm(
            enumerate(resampled_networks),
            total=resamples,
            unit="resample",
            bar_format="{percentage:3.0f}%|{bar:20}{r_bar}",
        )
        for resample, net in progress_bar:
            # print("Resample " + str(resample))
            val, model, net = retrain_net_NARMA(net[0], results_dict)
            _, t_performance = test_net_NARMA(net, model, error_margin, test_data)
            test_results.append(t_performance)

        save_path = path[:-2] + '_gen_' + str(maxgen) + '_test_optimized.p'
        print("Saving results to " + save_dir+save_path)
        with open(save_dir+save_path, 'wb') as f:
            pkl.dump(test_results, f)
