import numpy as np
import pickle as pkl
from reservoirpy import datasets
from simulator import NetworkSimulator
from utils import single_sample_NRSE, eval_candidate_signal_gen_horizon
from datetime import date
import argparse


def resample_net_MG_best(data_dict, maxgen=None):
    '''
    resample the best performing network from the evolutionary optimization, which can be used for retraining and testing on specific tau values, this can be used to test the performance of the best network on specific tau values and compare it to the performance of other networks or to the performance of the same network on different tau values

    
    :param data_dict: The dictionary containing the results of the evolutionary optimization, should contain the following keys: 'validation performance', 'parameters', and 'example net'
    :param maxgen: The maximum generation to consider in the evolutionary optimization results, if None, all generations are considered
    :return: The best performing network from the evolutionary optimization, should be a network object that can be used for retraining and testing on specific tau values  
    :rtype: Any

    '''





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
    '''
    return the worst performing network from the evolutionary optimization, which can be used for retraining and testing on specific tau values, this can be used to test the robustness of the evolutionary optimization results by comparing the performance of the best and worst networks on the same tau values

    :param data_dict: The dictionary containing the results of the evolutionary optimization, should contain the following keys: 'validation performance', 'parameters', and 'example net'      
    :return: The worst performing network from the evolutionary optimization, should be a network object that can be used for retraining and testing on specific tau values

    '''
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

def retrain_net_MG(best_net, data_dict, tau):       
    '''
    retrain the best network on the specific tau value, return the validation performance, 
    the model used for generating predictions from the network output, and the retrained network itself

    
    :param best_net: The best network to retrain, should be a network object that can be reset and updated, and should have the same architecture as the networks used in the evolutionary optimization 
    :param data_dict: The dictionary containing the results of the evolutionary optimization, should contain the following keys: 'tau range', 'start value range', 'error margin', 'alpha grid', and any other parameters needed for retraining the network
    :param tau: The tau value to retrain the network on, should be an integer that is within the tau range specified in the data dict
    
    :return: A tuple containing the validation performance of the retrained network, the model used for generating predictions from the network output, and the retrained network itself    

    
    '''

    best_net.reset_network()
    x0_range = data_dict['start value range']
    n_seq = data_dict['number of sequences']
    n_sam = data_dict['number of samples']
    tau_range = [tau, tau]
    error_margin = data_dict['error margin']
    alphas = data_dict['alpha grid']
    val, model, net = eval_candidate_signal_gen_horizon(best_net,
                                                          n_seq['unsupervised'],
                                                          n_seq['supervised'],
                                                          0,
                                                          n_sam['unsupervised'],
                                                          n_sam['supervised'],
                                                          0,
                                                          error_margin=error_margin,
                                                          alphas=alphas,
                                                          tau_range=tau_range,
                                                          x0_range=x0_range,
                                                          n_range=[10, 10]
                                                         )
    return val, model, net


def test_specific_net_from_evo(results_dict, gen, ind, network=None, n_test_sequences=5, n_test_samples=502, warmup=400, tau_list=None):
    '''
    Test the performance of a specific network from the evolutionary optimization results on a 
    specific set of sequences, return the predictions and the number of steps taken to reach the error 
    margin for each sequence      

    
    :param results_dict: The dictionary containing the results of the evolutionary optimization, should contain the following keys: 'tau list', 'start value range', 'error margin', 'example net', 'parameters'
    :param gen: The generation of the network to test, should be an integer between 0 and the number of generations in the results dict
    :param ind: The index of the network to test, should be an integer between 0 and the population size in the results dict
    :param network: The network to test, if None, the network will be reconstructed from the parameters in the results dict using the gen and ind arguments, if not None, the gen and ind arguments will be ignored
    :param n_test_sequences: The number of test sequences to generate for each tau, should be an integer, default is 5

    :param n_test_samples: The number of samples in each test sequence, should be an integer, default is 502
    :param warmup: The number of warmup steps to discard from each test sequence, should be an integer, default is 400
    :param tau_list: The list of tau values to test on, if None, the tau list from the results dict will be used, should be a list of integers, default is None

    :return: A dictionary containing the test results for each tau, where the keys are the tau values and the values are lists of the number of steps taken to reach the error margin for each test sequence

    '''
    if tau_list is None:
        tau_list = results_dict['tau list']  # get tau range from any of the results dict
    x0_range = results_dict['start value range']
    test_data_tau = {}
    for tau in tau_list:
        test_data = []
        for seq in range(n_test_sequences):
            test_sequence = datasets.mackey_glass(n_test_samples + warmup, tau=tau,
                                                  x0=np.random.uniform(x0_range[0], x0_range[1]))
            test_data.append(test_sequence)
        test_data_tau[tau] = test_data


    test_results = {}
    net_to_test = network
    if network is None:
        start_net = results_dict['example net']
        all_params = results_dict['parameters']
        specific_params = all_params[gen, ind, :]
        net_to_test = start_net.get_new_network_from_serialized(specific_params)

    unique_tau_list = list(set(tau_list)) # only go once through each tau
    for tau in unique_tau_list:
        test_results[tau] = []
        print("Testing for tau = " + str(tau))
        error_margin = results_dict['error margin']


        val, model, trained_net_to_test = retrain_net_MG(net_to_test, results_dict, tau)
        _, t_performance = test_net_MG(trained_net_to_test, model, error_margin, test_data_tau[tau])
        print(t_performance)
        test_results[tau].append(t_performance)

    return test_results




def test_net_MG(network, model, error_margin, test_data):
    '''Test the performance of a specific network on a specific set of sequences, 
    return the predictions and the number of steps taken to reach the error margin for each sequence  
    parameters:
    network: the network to test    
    model: the model to use for generating predictions from the network output
    error_margin: the error margin to use for determining when the network has successfully learned to predict the sequence
    test_data: a list of sequences to test the network on, each sequence should be a numpy array of shape (n_samples, 1)

    output:
    y_across_sequences: a list of lists, where each inner list contains the predictions made by the network for each sequence
    prediction_steps_across_sequences: a list of integers, where each integer represents the number of steps taken for the network to reach the error margin for each sequence
        

    '''
    warmup = 400
    prediction_steps_across_sequences = []
    y_across_sequences = []
    max_it_val = 500
    sim = NetworkSimulator(network)

    for sequence in test_data:
        start_input_val = sequence[warmup]
        labels_val = sequence[warmup + 1:]
        sim.warmup(sequence[:warmup])
        error = 0
        j = 0
        feedback_in = start_input_val
        label_variance = np.var(labels_val)
        y = []
        while j <= max_it_val and error < error_margin:
            feedback_in = np.ones((len(network.neurons_in),)) * feedback_in
            network.update_step(feedback_in)
            output = network.A[network.neurons_out, 0].T
            if len(output.shape) == 1:
                output = np.expand_dims(output, 0)
            feedback_in = model.predict(output)[0][0]
            y.append(feedback_in)
            error = single_sample_NRSE(feedback_in, labels_val[j, 0],
                                       label_variance)
            j += 1

        prediction_steps_across_sequences.append(j)
        y_across_sequences.append(y)
    return y_across_sequences, prediction_steps_across_sequences


def testVisualize(network, data):
    "'Visualize the performance of a specific network on a specific sequence"

    sim = NetworkSimulator(network, False)
    sim.visualize(data)



if __name__ == '__main__':
    '''Main function to test the performance of the best network from the evolutionary optimization on specific tau values, and save the test results to a file
    
    
    
    '''
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", action="store", type=str, default="./", help="Evolution results data path")
    parser.add_argument("-r", "--resamples", action="store", type=int, default=100, help="Number of resamples per test")
    parser.add_argument("-t", "--testsamples", action="store", type=int, default=502, help="Test sequence length")
    parser.add_argument("-s", "--testsequences", action="store", type=int, default=5, help="Number of test sequences per network")
    parser.add_argument("-g", "--maxgen", action="store", type=int, default=None, help="Takes best up to this generation")


    args = parser.parse_args()
    config = vars(args)
    resamples = config['resamples']
    n_test_samples = config['testsamples']
    n_test_sequences = config['testsequences']
    path = config['path']
    maxgen = config['maxgen']
    # Load data
    print("Loading hyperparameter optimization results from " + path)

    with open(path, 'rb') as f:
        results_dict = pkl.load(f)

    tau_range = results_dict['tau range'] # get tau range from any of the results dict
    tau_list = range(tau_range[0], tau_range[1] + 1)     # get tau list from any of the results dict
    x0_range = results_dict['start value range']

    # Generate test data
    # n_test_samples = 502
    test_data_tau = {}
    warmup = 400
    for tau in tau_list:
        test_data = []
        for seq in range(n_test_sequences):
            test_sequence = datasets.mackey_glass(n_test_samples + warmup, tau=tau,
                                              x0=np.random.uniform(x0_range[0], x0_range[1]))
            test_data.append(test_sequence)
        test_data_tau[tau] = test_data

    test_results = {}

    resampled_networks = []
    print("Sample networks")

    for resample in range(resamples):
        # best_net = resample_net_MG_worst(results_dict)
        # testVisualize(best_net, test_data_tau[14][0])
        best_net = resample_net_MG_best(results_dict, maxgen=maxgen)
        resampled_networks.append(best_net)

    unique_tau_list = list(set(tau_list)) # only go once through each tau
    for tau in unique_tau_list:
        test_results[tau] = []
        print("Testing for tau = " + str(tau))
        error_margin = results_dict['error margin']
        for resample, net in enumerate(resampled_networks):
            print("Resample " + str(resample))
            val, model, net = retrain_net_MG(net, results_dict, tau)
            _, t_performance = test_net_MG(net, model, error_margin, test_data_tau[tau])
            test_results[tau].append(t_performance)

    save_path = path[:-2] + '_gen' + str(maxgen) + '_test_optimized.p'
    with open(save_path, 'wb') as f:
        pkl.dump(test_results, f)