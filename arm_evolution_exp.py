from datetime import date, datetime
import argparse
import matplotlib.pyplot as plt
from future.backports.datetime import datetime
from reservoirpy.datasets import mackey_glass
from scipy.signal import lfilter, butter
from evolution import cmaes_IM
from populations import FlexiblePopulation, AdaptiveFlexiblePopulation
import numpy as np
from evolution import cmaes_multitask_narma
from simulator import NetworkSimulator, NetworkArmSimulator
from utils import createNARMA10, createNARMA30, inputs2NARMA
from config import propagation_vel, get_p_dict_heterogeneity_exp, get_p_dict_heterogeneity_exp_adaptive
import os
import pickle as pkl
from reservoirpy import datasets
from arm_model import Arm2Link

def get_FM_net(N, D):
    K = 1
    max_delay = D
    x_range = [-.01, .01]
    y_range = [-.01, .01]
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width ** 2 + height ** 2)
    max_time = max_dist / propagation_vel
    dt = max_time / max_delay
    size_in = 6
    size_out = 0
    in_loc = np.array([[x_range[1], y_range[0] + i * 0.1 * width] for i in range(size_in)])
    out_loc = None  # np.array([[x_range[0], y_range[1] - i * 0.1 * height] for i in range(size_out)])
    start_location_var = 0.002
    start_locatation_mean_var = 0.005
    start_weight_mean = .1
    start_weight_var = .1
    start_bias_mean = 0
    start_bias_var = .1
    c_start = np.array([.2])
    p_dict = get_p_dict_heterogeneity_exp_adaptive(K, x_range, y_range, start_location_var, start_locatation_mean_var,
                                                   start_weight_mean, start_weight_var, start_bias_mean, start_bias_var)
    p_dict['inhibitory']['val'] = np.array([.45])
    p_dict['in_connectivity']['val'] = np.array([.1])
    p_dict['in_mean']['val'] = np.array([.4])
    p_dict['connectivity']['val'] = c_start
    p_dict['in_connectivity']['val'] = np.array([.5])
    # p_dict['out_connectivity']['val'] = c_out
    # p_dict['theta0_scaling']['val'] = np.array([.4])
    # p_dict['feedback_mean']['val'] = np.array([1, 0])
    # p_dict['feedback_scaling']['val'] = np.array([.1, 0])
    # p_dict['feedback_connectivity']['val'] = c_feedback
    # p_dict['out_lr_mean']['val'] = np.array([.05])
    start_net = AdaptiveFlexiblePopulation(N, x_range, y_range, dt, in_loc, out_loc, size_in, size_out,
                                           p_dict)

    return start_net, dt

def get_IM_net(N, D):
    K = 4
    max_delay = D
    x_range = [-.01, .01]
    y_range = [-.01, .01]
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width ** 2 + height ** 2)
    max_time = max_dist / propagation_vel
    dt = max_time / max_delay
    size_in = 4
    size_out = 0
    in_loc = np.array([[x_range[1], y_range[0] + i * 0.1 * width] for i in range(size_in)])
    out_loc = None  # np.array([[x_range[0], y_range[1] - i * 0.1 * height] for i in range(size_out)])
    start_location_var = 0.002
    start_locatation_mean_var = 0.005
    start_weight_mean = .1
    start_weight_var = .1
    start_bias_mean = 0
    start_bias_var = .1
    c_start = np.array([.1])
    p_dict = get_p_dict_heterogeneity_exp_adaptive(K, x_range, y_range, start_location_var, start_locatation_mean_var,
                                                   start_weight_mean, start_weight_var, start_bias_mean, start_bias_var)
    p_dict['inhibitory']['val'] = np.array([.45])

    p_dict['in_mean']['val'] = np.array([.4])
    p_dict['connectivity']['val'] = c_start
    p_dict['in_connectivity']['val'] = np.array([.4])
    start_net = AdaptiveFlexiblePopulation(N, x_range, y_range, dt, in_loc, out_loc, size_in, size_out,
                                           p_dict)
    return start_net, dt

def get_torque_data_lpass(N, fs, cutoff, order, gain):
    torque_rand = np.random.uniform(-gain, gain, size=(N, 2))
    nyquist = 0.5 * fs # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    torque_rand = lfilter(b, a, torque_rand)
    return torque_rand

def get_torque_data_ma(N, window, gain):
    torque_rand = np.random.uniform(-gain, gain, size=(N, 2))
    t1 = moving_average(torque_rand[:, 0], window)
    t2 = moving_average(torque_rand[:, 1], window)
    return np.stack((t1, t2), axis=1)

def moving_average(data, window_size):
    kernel = np.ones((window_size)) / window_size
    return np.convolve(data, kernel, mode='valid')  # Use 'valid' to avoid edge effects


def get_IM_net_2_state(N, D):
    K = 4
    max_delay = D
    x_range = [-.01, .01]
    y_range = [-.01, .01]
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    max_dist = np.sqrt(width ** 2 + height ** 2)
    max_time = max_dist / propagation_vel
    dt = max_time / max_delay
    size_in = 8
    size_out = 0
    in_loc = np.array([[x_range[1], y_range[0] + i * 0.1 * width] for i in range(size_in)])
    out_loc = None  # np.array([[x_range[0], y_range[1] - i * 0.1 * height] for i in range(size_out)])
    start_location_var = 0.002
    start_locatation_mean_var = 0.005
    start_weight_mean = .1
    start_weight_var = .1
    start_bias_mean = 0
    start_bias_var = .1
    c_start = np.array([.1])
    p_dict = get_p_dict_heterogeneity_exp_adaptive(K, x_range, y_range, start_location_var, start_locatation_mean_var,
                                                   start_weight_mean, start_weight_var, start_bias_mean, start_bias_var)
    p_dict['inhibitory']['val'] = np.array([.45])

    p_dict['in_mean']['val'] = np.array([.4])
    p_dict['connectivity']['val'] = c_start
    p_dict['in_connectivity']['val'] = np.array([.4])
    # p_dict['out_connectivity']['val'] = c_out
    # p_dict['theta0_scaling']['val'] = np.array([.4])
    # p_dict['feedback_mean']['val'] = np.array([1, 0])
    # p_dict['feedback_scaling']['val'] = np.array([.1, 0])
    # p_dict['feedback_connectivity']['val'] = c_feedback
    # p_dict['out_lr_mean']['val'] = np.array([.05])
    start_net = AdaptiveFlexiblePopulation(N, x_range, y_range, dt, in_loc, out_loc, size_in, size_out,
                                           p_dict)
    return start_net, dt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--delay", action="store_true", help="Run experiment with delays")
    parser.add_argument("-nr", "--neurons", action="store", help="number of neurons", type=int, default=300)
    parser.add_argument("-s", "--suffix", action="store", help="filename suffix", type=str, default='')
    # parser.add_argument("-sd", "--seed", action="store", help="random seed", type=int, default=3)


    args = parser.parse_args()
    config = vars(args)
    delay = config['delay']
    # np.random.seed(4)
    D_max = .1
    net_type_str = "ESN"
    if delay:
        D_max = 15
        net_type_str = "DDN"

    N = config['neurons']
    filename_suffix = config['suffix']
    start_net, dt = get_IM_net(N, D_max)
    arm = Arm2Link()
    dt_arm = arm.dt
    gain = 1000

    fs = 1/dt_arm
    # torque_rand_train = get_torque_data_lpass(5000, fs, 1000, 1)
    # torque_rand_test = get_torque_data_lpass(5000, fs, 1000, 1)
    torque_rand_train = get_torque_data_ma(5000, 5, gain)
    torque_rand_test = get_torque_data_ma(5000, 5, gain)

    dir_name = 'arm_IM_evo'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = str(datetime.today()) + "_arm_IM_pilot_" + net_type_str + "_" + filename_suffix

    print('Experiment will be saved as')
    print(filename + '.pkl')

    max_it = 200
    pop_size = 20
    eval_reps = 4
    alphas = [10e-7, 10e-5, 10e-3, 10e-1]
    lag_grid = range(0, 10)
    warmup = 300
    cmaes_IM(start_net, torque_rand_train, torque_rand_test, max_it, pop_size, eval_reps, alphas, lag_grid,
             dir_name, filename, warmup)
