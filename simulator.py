import utils
from gui import DistDelayGUI, DistDelayGUI_arm
import numpy as np
import matplotlib.pyplot as plt
from populations import GMMPopulationAdaptive, AdaptiveFlexiblePopulation
from sklearn.linear_model import Ridge, RidgeCV
import network

class NetworkArmSimulator(object):

    def __init__(self, network, arm, plasticity=True, gain=100):
        self.network = network
        self.arm = arm
        self.plasticity = False
        self.gain = gain
        if type(self.network) is GMMPopulationAdaptive:
            self.plasticity = plasticity

    def scale_neuron2torque(self, activation):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        center_neuron = (range_neuron[0] + range_neuron[1]) / 2
        scale_neuron = (range_neuron[1] - range_neuron[0]) / 2
        torque = ((activation - center_neuron) / scale_neuron) * self.gain
        return torque

    def scale_torque2neuron(self, torque):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        center_neuron = (range_neuron[0] + range_neuron[1]) / 2
        scale_neuron = (range_neuron[1] - range_neuron[0]) / 2
        activation = (torque/self.gain) * scale_neuron + center_neuron
        return activation

    def scale_q2neuron(self, q):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        range_q = (0.1, 3.0)
        center_q = (range_q[0] + range_q[1])/2
        scale_q = (range_q[1] - range_q[0])/2
        center_neuron = (range_neuron[0] + range_neuron[1])/2
        scale_neuron = (range_neuron[1] - range_neuron[0])/2
        q = q - center_q
        q = (q / scale_q) * scale_neuron
        q += center_neuron
        return q

    def scale_dq2neuron(self, dq):
        act_func = self.network.activation_func
        range_neuron = (0, 1)
        scale_dq = self.gain/4
        if type(act_func) is network.tanh_activation:
            range_neuron = (-1, 1)
        range_dq = (-scale_dq, scale_dq)
        center_dq = (range_dq[0] + range_dq[1])/2
        # scale_dq = (range_dq[1] - range_dq[0])/2
        center_neuron = (range_neuron[0] + range_neuron[1])/2
        scale_neuron = (range_neuron[1] - range_neuron[0])/2
        dq = dq - center_dq
        dq = (dq / scale_dq) * scale_neuron
        dq += center_neuron
        return dq

    def sim_step(self, target):
        q = np.array(self.arm.q)
        dq = np.array(self.arm.dq)
        q = self.scale_q2neuron(q)
        dq = self.scale_dq2neuron(dq)
        net_in = np.concatenate([q, dq, target])
        self.network.update_step(net_in)
        net_out = self.network.A[self.network.neurons_out, 0]
        net_out = self.scale_neuron2torque(net_out, self.gain)
        self.arm.arm_func(net_out)

    def warmup(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)

    def unsupervised(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step_adaptive(inp)

    def get_network_data(self, input_data):
        net_out = []
        network_output_indices = self.network.neurons_out
        for i, inp in enumerate(input_data):
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)
            output = self.network.A[network_output_indices, 0]
            net_out.append(output)
        net_out = np.stack(net_out, axis=1)
        return net_out.reshape(net_out.shape[:2])

    def train_IM_simple(self, input_t, lag_grid, warmup=50, alphas=[10e-7, 10e-5, 10e-3, 10e-1]):
        assert len(self.network.neurons_in) == 4  # q, dq
        network_states = []
        labels = []
        for i, torque in enumerate(input_t):
            data = self.arm.arm_func(torque)
            labels.append(torque)
            q = self.scale_q2neuron(data[:2])
            dq = self.scale_q2neuron(data[2:4])
            inp = np.concatenate([q, dq])
            act = self.network.update_step(inp)
            network_states.append(act)

        model_per_lag = {}
        s_per_lag = {}
        network_states_clipped = network_states[warmup:]
        labels_clipped = labels[warmup:]
        for l in lag_grid:
            if l > 0:
                labels_clipped = labels[warmup - l: -l]

            model = RidgeCV(alphas=alphas, cv=5)
            model.fit(network_states_clipped, labels_clipped)
            model_per_lag[l] = model
            s_per_lag[l] = model.best_score_
        best_lag = max(s_per_lag, key=s_per_lag.get)
        self.model = model_per_lag[best_lag]
        self.lag = best_lag

    def test_IM_simple(self, input_t, warmup=50):
        assert len(self.network.neurons_in) == 4
        assert not self.model is None
        assert not self.lag is None
        labels = []
        predictions = []
        for i, torque in enumerate(input_t):
            data = self.arm.arm_func(torque)
            labels.append(torque)
            q = self.scale_q2neuron(data[:2])
            dq = self.scale_q2neuron(data[2:4])
            inp = np.concatenate([q, dq])
            act = self.network.update_step(inp)
            prediction = self.model.predict(act.reshape(1, -1))[0]
            predictions.append(prediction)
        labels_clipped = labels[warmup:]
        predictions_clipped = predictions[warmup:]
        if self.lag > 0:
            labels_clipped = labels[warmup - self.lag:-self.lag]

        return utils.nrmse(np.array(labels_clipped), np.array(predictions_clipped))

    def train_IM(self, training_torque, n_states_past, warmup=50):
        assert len(self.network.neurons_in) == 8 # q1, dq1, q2, dq2
        network_states = []
        labels = []
        past_arm_states = [np.array([0, 0, 0, 0]) for _ in range(n_states_past)]
        past_torque_states = [np.array([0, 0]) for _ in range(n_states_past)]
        for i, torque in enumerate(training_torque):
            past_torque_states.append(torque)
            labels.append(past_torque_states.pop(0))
            # apply torque to arm
            data = self.arm.arm_func(torque)
            q_now = self.scale_q2neuron(data[:2])
            dq_now = self.scale_dq2neuron(data[2:4])
            state_now = np.concatenate([q_now, dq_now])
            state_past = past_arm_states.pop(0)
            past_arm_states.append(state_now) # add q and dq
            inp = np.concatenate([state_past, state_now])
            act = self.network.update_step(inp)
            network_states.append(act)

        lag_grid = range(0, 15)
        model_per_lag = {}
        s_per_lag = {}
        network_states_clipped = network_states[warmup:]
        labels_clipped = labels[warmup:]
        for l in lag_grid:
            if l > 0:
                labels_clipped = labels[warmup -l: -l]

            model = RidgeCV(alphas=[10e-7, 10e-5, 10e-3], cv=5)
            model.fit(network_states_clipped, labels_clipped)
            model_per_lag[l] = model
            s_per_lag[l] = model.best_score_
        best_lag = max(s_per_lag, key=s_per_lag.get)
        self.model = model_per_lag[best_lag]
        self.lag = best_lag

    def train_FM(self, random_torque, n_steps_ahead, warmup=50):
        assert len(self.network.neurons_in) == 6 # q, dq, u
        network_states = []
        labels = []
        for i, torque in enumerate(random_torque):
            # apply torque to arm
            data = self.arm.arm_func(torque)
            inp_t = self.scale_torque2neuron(torque)
            inp_q = self.scale_q2neuron(data[:2])
            inp_dq = self.scale_dq2neuron(data[2:4])
            inp = np.concatenate([inp_t, inp_q, inp_dq])
            act = self.network.update_step(inp)
            network_states.append(act)
            labels.append(data[:4])
        network_states = network_states[warmup-n_steps_ahead:-n_steps_ahead]
        labels = labels[warmup:]
        self.model = RidgeCV(alphas=[10e-7, 10e-5, 10e-3], cv=5)
        self.model.fit(network_states, labels)

    def test_FM(self, random_torque, n_steps_ahead, warmup=50):
        assert len(self.network.neurons_in) == 6  # q, dq, u
        predictions = []
        labels = []
        for i, torque in enumerate(random_torque):
            # apply torque to arm
            data = self.arm.arm_func(torque)
            inp_t = self.scale_torque2neuron(torque)
            inp_q = self.scale_q2neuron(data[:2])
            inp_dq = self.scale_dq2neuron(data[2:4])
            inp = np.concatenate([inp_t, inp_q, inp_dq])
            act = self.network.update_step(inp)
            prediction = self.model.predict(act.reshape(1, -1))[0]
            predictions.append(prediction)
            labels.append(data[:4])
        predictions_clipped = predictions[warmup - n_steps_ahead:-n_steps_ahead]
        labels_clipped = labels[warmup:]
        return utils.nrmse(np.array(predictions_clipped), np.array(labels_clipped))

    def visualize_FM(self, random_torque, lag=1):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        lagged_prediction = [np.array([0, 0]) for _ in range(lag)]
        leg = ['u1', 'u2', 'q1', 'q2', 'dq1', 'dq2']
        label_str = "Best score: " + str(self.model.best_score_)

        for i, torque in enumerate(random_torque):
            # apply torque to arm
            data = self.arm.arm_func(torque)

            new_q = data[:2]
            error = (lagged_prediction.pop(0) - new_q)

            inp_t = self.scale_torque2neuron(torque)
            inp_q = self.scale_q2neuron(data[:2])
            inp_dq = self.scale_dq2neuron(data[2:4])
            inp = np.concatenate([inp_t, inp_q, inp_dq])
            net_act = self.network.update_step(inp)
            prediction = self.model.predict(net_act.reshape(1, -1))
            lagged_prediction.append(prediction[0, :2])
            gui.update_a(self.arm, fm_prediction=prediction, p_error=error, error_scale=.25, legend=leg,
                         debug_label=label_str)
        gui.close()

    def visualize_FM_dynamic_input(self, t_input, lag=1):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        lagged_prediction = [np.array([0, 0]) for i in range(lag)]
        leg = ['u1', 'u2', 'q1', 'q2', 'dq1', 'dq2']

        # middles = (self.arm.q_upper_limits + self.arm.q_lower_limits) / 2
        for i, torque in enumerate(t_input):

            arm_state = self.arm.q
            # dist_to_middle = middles - arm_state
            # torque = gain * dist_to_middle

            dist_to_upper = self.arm.q_upper_limits - arm_state
            dist_to_lower = arm_state - self.arm.q_lower_limits
            torque = torque + self.gain * (1 / dist_to_lower - 1 / dist_to_upper)# + np.random.uniform(-1, 1, size=(2,)))
            # apply torque to arm
            data = self.arm.arm_func(torque)

            new_q = data[:2]

            error = (lagged_prediction.pop(0) - new_q)

            inp_t = self.scale_torque2neuron(torque)
            inp_q = self.scale_q2neuron(data[:2])
            inp_dq = self.scale_dq2neuron(data[2:4])
            inp = np.concatenate([inp_t, inp_q, inp_dq])
            net_act = self.network.update_step(inp)
            prediction = self.model.predict(net_act.reshape(1, -1))
            lagged_prediction.append(prediction)
            gui.update_a(self.arm, fm_prediction=prediction, p_error=error, error_scale=.25, legend=leg)
        gui.close()

    def visualize_IM_simple(self, random_torque, gain=2000):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        past_torque_states = [[0, 0] for i in range(self.lag)]
        leg = ['q1', 'q2', 'dq1', 'dq2']
        label_str = "Best score: " + str(self.model.best_score_) + ", Best lag: " + str(self.lag)
        for i, torque in enumerate(random_torque):
            # apply torque to arm
            arm_state = self.arm.q

            dist_to_upper = self.arm.q_upper_limits - arm_state
            dist_to_lower = arm_state - self.arm.q_lower_limits
            dist_to_lower = 1/(dist_to_lower + 0.00001)
            dist_to_upper = 1/(dist_to_upper + 0.00001)
            torque = torque + self.gain/2 * (dist_to_lower - dist_to_upper)# + np.random.uniform(-1, 1, size=(2,)))

            past_torque_states.append(torque)
            data = self.arm.arm_func(torque)
            q = self.scale_q2neuron(data[:2])
            dq = self.scale_dq2neuron(data[2:4])
            inp = np.concatenate([q, dq])
            net_act = self.network.update_step(inp)
            prediction = self.model.predict(net_act.reshape(1, -1))
            error = (prediction - past_torque_states.pop(0))
            gui.update_a(self.arm, p_error=error, error_scale=gain, legend=leg, debug_label=label_str)
        gui.close()

    def visualize_IM(self, random_torque, n_states_past=1):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        past_arm_states = [[0, 0, 0, 0] for i in range(n_states_past)]
        past_torque_states = [[0, 0] for i in range(self.lag + 1)]
        leg = ['q1_past', 'q2_past', 'dq1_past', 'dq2_past', 'q1_now', 'q2_now', 'dq1_now', 'dq2_now']
        label_str = "Best score: " + str(self.model.best_score_) + ", Best lag: " + str(self.lag)
        for i, torque in enumerate(random_torque):
            # apply torque to arm
            data = self.arm.arm_func(torque)

            inp_state_1 = past_arm_states.pop(0)
            inp_q2 = self.scale_q2neuron(data[:2])
            inp_dq2 = self.scale_dq2neuron(data[2:4])
            inp_state_2 = np.concatenate([inp_q2, inp_dq2])
            past_arm_states.append(inp_state_2)

            inp = np.concatenate([inp_state_1, inp_state_2])
            net_act = self.network.update_step(inp)
            prediction = self.model.predict(net_act.reshape(1, -1))
            error = (prediction - past_torque_states.pop(0))
            past_torque_states.append(torque)
            gui.update_a(self.arm, p_error=error, error_scale=self.gain, legend=leg, debug_label=label_str)
        gui.close()

    def visualize_arm(self, random_torque, target):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        j = 0
        for i, torque in enumerate(random_torque):
            # apply torque to arm
            data = self.arm.arm_func(torque)
            j += 1
            inp_t = self.scale_torque2neuron(torque)
            inp_q = self.scale_q2neuron(data[:2])
            inp_dq = self.scale_dq2neuron(data[2:4])
            inp = np.concatenate([inp_t, inp_q, inp_dq])
            if self.plasticity:
                self.network.update_step_adaptive(inp)
            else:
                self.network.update_step(inp)

            gui.update_a(self.arm, target)
        gui.close()

    def visualize(self, input_data):
        gui = DistDelayGUI(self.network)
        j = 0

        for i, input in enumerate(input_data):
            j += 1
            inp = np.ones((len(self.network.neurons_in),)) * input
            if self.plasticity:
                self.network.update_step_adaptive(inp)
            else:
                self.network.update_step(inp)

            gui.update_a()
            # if j == 100:
            #     j = 0
            #     if self.network.buffersize > 1:
            #         Wds = np.array(self.network.W_masked_list)
            #         newW = np.sum(Wds, axis=0)
            #     else:
            #         newW = self.network.W
            #     nonzeroNewW = list(np.reshape(newW, newW.shape[0] * newW.shape[1]))
            #     nonzeroNewW = [i for i in nonzeroNewW if i != 0]
            #     print('min (nonzero) ', np.min(nonzeroNewW))
            #     print('max (nonzero) ', np.max(nonzeroNewW))
            #     print('average (nonzero) ', np.average(nonzeroNewW))
            #     print('std (nonzero) ', np.std(nonzeroNewW))
            #     print('N nonzero weights ', len(nonzeroNewW))
            #     print('norm', np.linalg.norm(newW))
            #     # plt.hist(nonzeroNewW, bins=20)
            #     # plt.pause(0.01)
        gui.close()

    def reset(self):
        self.network.reset_activity()

    def full_reset(self):
        self.network.reset_network()


class NetworkSimulator(object):

    def __init__(self, network, plasticity=True):
        self.network = network
        self.plasticity = False
        if type(self.network) is GMMPopulationAdaptive or type(self.network) is AdaptiveFlexiblePopulation:
            self.plasticity = plasticity

    def warmup(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)

    def unsupervised(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step_adaptive(inp)

    def get_network_data(self, input_data):
        net_out = []
        network_output_indices = self.network.neurons_out
        for i, inp in enumerate(input_data):
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)
            output = self.network.A[network_output_indices, 0]
            net_out.append(output)
        net_out = np.stack(net_out, axis=1)
        return net_out.reshape(net_out.shape[:2])

    def get_next_step(self, single_input):
        network_output_indices = self.network.neurons_out
        inp = np.ones((len(self.network.neurons_in),)) * single_input
        self.network.update_step(inp)
        output = self.network.A[network_output_indices, 0]
        return output

    def visualize_feedback(self, start_value, readout_model, labels):
        gui = DistDelayGUI(self.network)
        inp = start_value
        network_output_indices = self.network.neurons_out

        for l in labels:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)
            output = self.network.A[network_output_indices, 0].T
            inp = readout_model.predict(output)[0,0]
            abs_error = np.abs(l - inp)
            print('Absolute error:', abs_error)
            gui.update_a()

    def visualize(self, input_data):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        j = 0

        for i, input in enumerate(input_data):
            j += 1
            inp = np.ones((len(self.network.neurons_in),)) * input
            if self.plasticity:
                self.network.update_step_adaptive(inp)
            else:
                self.network.update_step(inp)

            gui.update_a()
            # if j == 100:
            #     j = 0
            #     if self.network.buffersize > 1:
            #         Wds = np.array(self.network.W_masked_list)
            #         newW = np.sum(Wds, axis=0)
            #     else:
            #         newW = self.network.W
            #     nonzeroNewW = list(np.reshape(newW, newW.shape[0] * newW.shape[1]))
            #     nonzeroNewW = [i for i in nonzeroNewW if i != 0]
            #     print('min (nonzero) ', np.min(nonzeroNewW))
            #     print('max (nonzero) ', np.max(nonzeroNewW))
            #     print('average (nonzero) ', np.average(nonzeroNewW))
            #     print('std (nonzero) ', np.std(nonzeroNewW))
            #     print('N nonzero weights ', len(nonzeroNewW))
            #     print('norm', np.linalg.norm(newW))
            #     # plt.hist(nonzeroNewW, bins=20)
            #     # plt.pause(0.01)
        gui.close()

    def visualize_arm(self, input_data, arm, target):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        j = 0

        for i, input in enumerate(input_data):
            j += 1
            inp = np.ones((len(self.network.neurons_in),)) * input
            if self.plasticity:
                self.network.update_step_adaptive(inp)
            else:
                self.network.update_step(inp)

            gui.update_a(arm, target)
        gui.close()

    def visualize_clamped(self, input_data, clamped_output):
        gui = DistDelayGUI(self.network, use_ntypes=False)
        j = 0

        for i, input in enumerate(input_data):
            j += 1
            inp = np.ones((len(self.network.neurons_in),)) * input
            if self.plasticity:
                self.network.update_step_adaptive(inp, clamped_output[i])
            else:
                self.network.update_step(inp, clamped_output[i])

            gui.update_a()
            # if j == 100:
            #     j = 0
            #     if self.network.buffersize > 1:
            #         Wds = np.array(self.network.W_masked_list)
            #         newW = np.sum(Wds, axis=0)
            #     else:
            #         newW = self.network.W
            #     nonzeroNewW = list(np.reshape(newW, newW.shape[0] * newW.shape[1]))
            #     nonzeroNewW = [i for i in nonzeroNewW if i != 0]
            #     print('min (nonzero) ', np.min(nonzeroNewW))
            #     print('max (nonzero) ', np.max(nonzeroNewW))
            #     print('average (nonzero) ', np.average(nonzeroNewW))
            #     print('std (nonzero) ', np.std(nonzeroNewW))
            #     print('N nonzero weights ', len(nonzeroNewW))
            #     print('norm', np.linalg.norm(newW))
            #     # plt.hist(nonzeroNewW, bins=20)
            #     # plt.pause(0.01)
        gui.close()

    def reset(self):
        self.network.reset_activity()

    def full_reset(self):
        self.network.reset_network()
