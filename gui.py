import tkinter as tk
from tkinter.constants import HORIZONTAL

from PIL import Image, ImageTk
import numpy as np
from skimage.draw import line_aa, disk
import config
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def get_tk_im(array):
    array_im = np.asarray(array * 255, dtype='uint8')
    pilim = Image.fromarray(array_im)
    img = ImageTk.PhotoImage(image=pilim)
    return img


def draw_line(array_pos, array_neg, coord_start, coord_end, weight):
    """
    Draws a line on a np array.
    :param array: ndarray
        Image array to draw a line on.
    :param coord_start: (int, int)
        x and y position of start of line.
    :param coord_end: (int, int)
        x and y position of end of line.
    :return: ndarray
        Original image array with line drawn.
    """
    x1, y1 = coord_start
    x2, y2 = coord_end
    assert array_pos.shape[0] > x1 and array_pos.shape[0] > x2 and array_pos.shape[1] > y1 \
           and array_pos.shape[1] > y2, 'coordinates not in range '
    assert x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0, 'Coordinates should be non-negative'
    assert array_pos.shape == array_neg.shape
    rr, cc, val = line_aa(x1, y1, x2, y2)
    if weight >= 0:
        array_pos[rr, cc] = 1 - val * weight * 2
    else:
        array_neg[rr, cc] = 1 - val * -weight * 2
    return array_pos, array_neg


class DistDelayGUI(object):
    """
    Animation for the distance-based delay network
    """

    def __init__(self, dist_delay_net, use_ntypes=False):
        self.DDN = dist_delay_net

        x_range = self.DDN.x_range
        y_range = self.DDN.y_range
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]
        window_width = 300

        self.spacing = window_width/width
        self.dot_size = 5
        self.use_ntypes = use_ntypes
        self.w = int(window_width + 2*self.dot_size + 2)
        self.h = int(window_width * height/width + 2*self.dot_size + 2)
        self.root = tk.Tk()
        self.conn_hist = np.zeros(shape=(100,))
        self.energy_use = np.zeros(shape=(100,))
        self.out_hist = np.zeros(shape=(100,))
        self.error_hist = np.zeros(shape=(100,))
        self.fm_error_hist = np.zeros(shape=(100,2))
        self.error_hist_av = np.zeros(shape=(100,))
        self.inner_hist = np.zeros(shape=(100,))
        self.real_out_hist = np.zeros(shape=(100,))
        self.in_hist = np.zeros(shape=(100,self.DDN.size_in))
        self.time_axis_plots_ms = np.arange(0, 100, 1) * self.DDN.dt * 1000

        grid = self.DDN.coordinates
        # shift and scale grid
        grid[:, 0] -= np.min(grid[:, 0])
        grid[:, 1] -= np.min(grid[:, 1])
        self.grid = grid

        self.frame1 = tk.Frame(self.root)
        self.frame1.pack(side=tk.LEFT)

        self.frame2 = tk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT)

        # self.frame3 = tk.Frame(self.root)
        # self.frame3.pack(side=tk.LEFT)

        self.frame4 = tk.Frame(self.root)
        self.frame4.pack(side=tk.RIGHT)

        self.scale_base = self.draw_scale()

        self.canvas = tk.Canvas(self.frame1, width=self.h + 100, height=self.w + 100)
        self.canvas2 = tk.Canvas(self.frame1, width=self.h + 100, height=self.w + 100)
        self.canvas.grid(row=1, column=0)
        self.canvas2.grid(row=2, column=0)
        # self.canvas.create_text(100, self.h- 100, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))

        # self.canvas2 = tk.Canvas(self.frame3, width=self.h + 50, height=self.w + 50)
        # self.canvas2.grid(row=1, column=0)


        fig, axs = plt.subplots(2, figsize=(4, 4))
        fig.tight_layout(pad=3)
        self.ax = axs
        self.canvas3 = FigureCanvasTkAgg(fig, master=self.frame4)
        self.canvas3.get_tk_widget().grid(row=1, column=1)

        def connection_update_callback():
            self.reset_connections()

        self.debug_label = tk.Label(self.frame1, text="", font=("Helvetica", 15))
        self.debug_label.grid(row=0, column=0)

        self.button = tk.Button(self.frame2, text="Update connection visualization", command=connection_update_callback)
        self.button.grid(row=0, column=1)
        self.lr_slider = tk.Scale(self.frame2, orient=HORIZONTAL, from_=0, to=1, resolution=.01)
        self.lr_slider.grid(row=1, column=1)
        self.lr_label = tk.Label(self.frame2, text="scale learning rate", font=("Helvetica", 10))
        self.lr_label.grid(row=1, column=2)
        self.lr_slider.set(0)
        self.clamp_slider = tk.Scale(self.frame2, orient=HORIZONTAL, from_=0, to=1, resolution=.01)
        self.clamp_slider.grid(row=2, column=1)
        self.clamp_slider.set(0)
        self.clamp_label = tk.Label(self.frame2, text="clamp output", font=("Helvetica", 10))
        self.clamp_label.grid(row=2, column=2)
        self.set_connections()
        # self.max_connectivity = np.product(self.DDN.connectivity.shape)

    def update_sliders(self):
        lr_val = self.lr_slider.get()
        clamp_val = self.clamp_slider.get()
        self.DDN.clamp_scale = clamp_val
        self.DDN.lr_scale = lr_val

    def update_debug(self, text):
        if not text is None:
            self.debug_label.config(text=text)

    def update_plots(self, p_error, error_scale, legend):
        # self.out_plot(self.ax[0])
        # self.error_plot(self.ax[0])
        self.ax[0].clear()
        self.ax[1].clear()
        self.in_plot_arm(self.ax[0], legend)
        # self.inner_plot(self.ax[1])
        self.p_error_plot(self.ax[1], p_error, error_scale)
        # self.weight_hist_plot(self.ax[1])
        self.canvas3.draw()

    def update_arm(self, arm, target=None, fm_prediction=None):
        w = self.w
        h = int(self.h)
        s = 50
        arm_base = np.zeros((w, h, 3))
        x, y = arm.position()
        p0 = (int(w/2), int(h/2))
        p1 = (p0[0] + int(x[1] * s), p0[1] + int(y[1] * s))
        p2 = (p0[0] + int(x[2] * s), p0[1] + int(y[2] * s))

        rr, cc = disk((p2[0], p2[1]), self.dot_size)
        arm_base[rr, cc, :] = 1

        rr, cc, val = line_aa(p0[0], p0[1], p1[0], p1[1])
        arm_base[rr, cc, :] = np.array([val, val, val]).T
        rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1])
        arm_base[rr, cc, :] = np.array([val, val, val]).T

        if target is not None:
            xt, yt = target
            t = (p0[0] + int(xt * s), p0[1] + int(yt * s))
            rr, cc = disk((t[0], t[1]), self.dot_size)
            arm_base[rr, cc, 0] = 1

        if fm_prediction is not None:
            x_pred, y_pred = arm.position([fm_prediction[0, 0], fm_prediction[0, 1]])
            p1_fm = (p0[0] + int(x_pred[1] * s), p0[1] + int(y_pred[1] * s))
            p2_fm = (p0[0] + int(x_pred[2] * s), p0[1] + int(y_pred[2] * s))

            rr, cc = disk((p2_fm[0], p2_fm[1]), self.dot_size)
            arm_base[rr, cc, 0] = 1

            rr, cc, val = line_aa(p0[0], p0[1], p1_fm[0], p1_fm[1])
            arm_base[rr, cc, :] = np.array([val, val, val]).T
            rr, cc, val = line_aa(p1_fm[0], p1_fm[1], p2_fm[0], p2_fm[1])
            arm_base[rr, cc, :] = np.array([val, val, val]).T

        return arm_base


    def weight_hist_plot(self, ax, bins=50):
        ax.hist(self.updated_weights_flattened, bins=bins)
        ax.set_title("Updated weight distribution")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")

    def in_plot_arm(self, ax, arm_legend=None):
        ax.plot(self.time_axis_plots_ms, self.in_hist)
        ax.set_ylim(-.1, 1.1)
        ax.set_title('Network input')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        if arm_legend is not None:
            ax.legend(arm_legend, loc='upper left', fontsize='small')

        self.in_hist[:-1] = self.in_hist[1:]
        self.in_hist[-1] = self.DDN.A[self.DDN.neurons_in, 0]

    def out_plot(self, ax):
        # self.out_hist = self.out_hist/((np.max(self.out_hist) - np.min(self.out_hist)) + .00001)
        ax.plot(self.time_axis_plots_ms, self.out_hist)
        ax.plot(self.time_axis_plots_ms, self.in_hist)
        ax.plot(self.time_axis_plots_ms, self.real_out_hist)
        ax.set_ylim(0, 2)
        ax.set_title('Network Output & input')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        self.out_hist[:-1] = self.out_hist[1:]
        self.in_hist[:-1] = self.in_hist[1:]
        self.real_out_hist[:-1] = self.real_out_hist[1:]
        self.out_hist[-1] = self.DDN.A[self.DDN.neurons_out, 0]
        self.real_out_hist[-1] = self.DDN.real_out
        self.in_hist[-1] = self.DDN.A[self.DDN.neurons_in, 0]


    def inner_plot(self, ax):
        ax.plot(self.time_axis_plots_ms, self.inner_hist)
        ax.set_ylim(0, 100)
        ax.set_title('Activity inner product')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Inner product activity')
        self.inner_hist[:-1] = self.inner_hist[1:]
        act = self.DDN.A[self.DDN.neurons_res, 0]
        self.inner_hist[-1] = act @ act.T

    def p_error_plot(self, ax, error, scale):
        ax.plot(self.time_axis_plots_ms, self.fm_error_hist)
        ax.set_ylim(-scale, scale)
        ax.set_title('Prediction Error')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Absolute error')
        self.fm_error_hist[:-1] = self.fm_error_hist[1:]
        self.fm_error_hist[-1] = error

    def error_plot(self, ax):
        ax.plot(self.time_axis_plots_ms, self.error_hist)
        ax.plot(self.time_axis_plots_ms, self.error_hist_av)

        ax.set_ylim(0, 100)
        ax.set_title('Prediction Error')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Absolute error')
        ax.legend(['Error', 'Time Window Average'])
        self.error_hist[:-1] = self.error_hist[1:]
        self.error_hist[-1] = np.abs(self.DDN.out_error)


        self.error_hist_av[:-1] = self.error_hist_av[1:]
        self.error_hist_av[-1] = np.abs(self.DDN.error_average)
    def energy_plot(self, ax):
        ax.plot(self.time_axis_plots_ms, self.energy_use)
        ax.set_ylim(0, 10)
        ax.set_title('Network power')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (W)')
        self.energy_use[:-1] = self.energy_use[1:]
        self.energy_use[-1] = self.DDN.get_current_power_bio()

    def set_connections(self):
        weights = self.DDN.W
        self.updated_weights = weights
        self.updated_weights_flattened = np.reshape(self.updated_weights, (np.product(self.updated_weights.shape),))
        self.connections_pos = None
        self.connections_neg = None
        self.get_connection_base(weights)
        dot_ex, dot_in = self.grid2dots(True)
        all_d = np.stack([dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base], axis=-1)
        connections_all = self.connections_pos + self.connections_neg
        self.img_c = np.stack([self.connections_pos, connections_all, self.connections_neg], axis=-1) * \
                     (all_d == 0)

    def reset_connections(self):
        self.updated_weights = sum(self.DDN.W_masked_list)
        self.updated_weights_flattened = np.reshape(self.updated_weights, (np.product(self.updated_weights.shape),))
        self.connections_pos = None
        self.connections_neg = None
        self.get_connection_base(self.updated_weights)
        dot_ex, dot_in = self.grid2dots(True)
        all_d = np.stack([dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base], axis=-1)
        connections_all = self.connections_pos + self.connections_neg
        self.img_c = np.stack([self.connections_pos, connections_all, self.connections_neg], axis=-1) * \
                     (all_d == 0)

    def get_connection_base(self, weights):
        """
        Generates an image of the non-zero connections in this network.
        :return: None
        """

        # weights = np.asarray(weights > 0, dtype='uint8')
        grid = self.grid

        N = grid.shape[0]
        connections_pos = np.ones((self.w, self.h))
        connections_neg = np.ones((self.w, self.h))
        for n1 in range(N):
            for n2 in range(N):
                if weights[n1, n2] != 0:
                    x1 = int(grid[n1, 0] * self.spacing + self.dot_size)
                    y1 = int(grid[n1, 1] * self.spacing + self.dot_size)
                    x2 = int(grid[n2, 0] * self.spacing + self.dot_size)
                    y2 = int(grid[n2, 1] * self.spacing + self.dot_size)
                    connections_pos, connections_neg = draw_line(connections_pos, connections_neg, (x1, y1), (x2, y2),
                                                                 weights[n1, n2])
                    # xa1 = x2 + 0.95 * (x1 - x2) + 5
                    # ya1 = y2 + 0.95 * (y1 - y2) + 5
                    # xa2 = x2 + 0.95 * (x1 - x2) - 5
                    # ya2 = y2 + 0.95 * (y1 - y2) - 5
                    # connections = draw_line(connections, (int(xa1), int(ya1)), (x2, y2))
                    # connections = draw_line(connections, (int(xa2), int(ya2)), (x2, y2))
        self.connections_pos = connections_pos
        self.connections_neg = connections_neg

    def draw_scale(self):
        # compute how many dts fit in a quarter screen
        width = self.h
        dt_width = self.DDN.dt * config.propagation_vel * self.spacing
        n_dt = int(np.floor(.45 * width/dt_width))

        scale_base = np.zeros((self.w, self.h))
        start_pos = np.array([self.w - 30, 20])
        end_pos = start_pos + np.array([0, n_dt * dt_width])
        # main line
        rr, cc, val = line_aa(int(start_pos[0]), int(start_pos[1]), int(end_pos[0]), int(end_pos[1]))
        scale_base[rr, cc] = val
        # vertical markers
        rr, cc, val = line_aa(int(start_pos[0] + 8), int(start_pos[1]), int(start_pos[0] - 8), int(start_pos[1]))
        scale_base[rr, cc] = val
        rr, cc, val = line_aa(int(end_pos[0] + 8), int(end_pos[1]), int(end_pos[0] - 8), int(end_pos[1]))
        scale_base[rr, cc] = val

        for i in range(n_dt):
            rr, cc, val = line_aa(int(start_pos[0] + 5), int(start_pos[1] + i * dt_width),
                                  int(start_pos[0] - 5), int(start_pos[1] + i * dt_width))
            scale_base[rr, cc] = val
        return scale_base


    def grid2dots(self, init=False):
        """
        Generates an image as a ndarray with neuron activation drawn as dots according to coordinates attribute.
        :param init: bool
            Sets activity of all neurons to 1. Only used when initial connection base visualisation is being
            created.
        :return: ndarray, ndarray
            Two w by h array with excitatory and inhibitory neuron activation respectively as dots drawn in the
            array.
        """
        coordinates = self.grid
        act = self.DDN.A[:, 0] #* self.DDN.ex_in
        if self.use_ntypes:
            act *= self.DDN.n_type
        # act = self.DDN.V
        # lower = -71
        # upper = np.average(self.DDN.V_threshold)
        # center = (upper + lower)/2
        # scale = (upper - lower)/2
        # act = (act - center + .5) / scale
        # act = np.clip(act, 0, 1) * self.DDN.ex_in

        if init:
            act = np.ones_like(act)
        N = coordinates.shape[0]
        dots_ex = np.zeros((self.w, self.h))
        dots_in = np.zeros((self.w, self.h))
        for i in range(N):
            a = act[i]
            x = np.round(coordinates[i, 0] * self.spacing + self.dot_size)
            y = np.round(coordinates[i, 1] * self.spacing + self.dot_size)
            rr, cc = disk((x, y), self.dot_size)

            if a > 0:
                dots_ex[rr, cc] = a
                dots_in[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 0
                dots_in[rr, cc] = -a
        return dots_ex, dots_in

    def update_a(self, arm=None, target=None, fm_prediction=None, p_error=None, error_scale=1, legend=None, debug_label=None):
        """
        Updates the gui according to current neuron activity.
        :return: None
        """
        dot_ex, dot_in = self.grid2dots()
        zer = np.zeros_like(self.connections_pos)
        img_d = np.stack([dot_in, dot_ex, zer], axis=-1)
        img = self.img_c + img_d
        img = get_tk_im(img)
        if not (arm is None):
            img_arm = np.transpose(self.update_arm(arm, target, fm_prediction), [1, 0, 2])
            img_arm = get_tk_im(img_arm)

            self.canvas2.create_image(3, 3, anchor="nw", image=img_arm)
        self.update_plots(p_error, error_scale, legend)
        self.canvas.create_image(5, 5, anchor="nw", image=img)
        self.update_sliders()
        # self.canvas2.create_image(10, 40, anchor='nw', image=img_conn)
        self.root.update()
        self.update_debug(debug_label)

    def get_single_frame(self):
        dot_ex, dot_in = self.grid2dots()
        zer = np.zeros_like(self.connections_pos)
        img_d = np.stack([dot_in, dot_ex, zer], axis=-1)
        img = self.img_c + img_d
        # img = get_tk_im(img)
        return img

    def close(self):
        self.root.destroy()

class DistDelayGUI_arm(object):
    """
    Animation for the distance-based delay network
    """

    def __init__(self, dist_delay_net, arm):
        self.DDN = dist_delay_net
        self.arm = arm

        x_range = self.DDN.x_range
        y_range = self.DDN.y_range
        width = x_range[1] - x_range[0]
        height = y_range[1] - y_range[0]
        window_width = 500

        self.spacing = window_width/width
        self.dot_size = 5
        self.w = int(window_width + 2*self.dot_size + 2)
        self.h = int(window_width * height/width + 2*self.dot_size + 2)
        self.root = tk.Tk()
        self.conn_hist = np.zeros(shape=(100,))
        self.energy_use = np.zeros(shape=(100,))
        self.time_axis_plots_ms = np.arange(0, 100, 1) * self.DDN.dt * 1000

        grid = self.DDN.coordinates
        # shift and scale grid
        grid[:, 0] -= np.min(grid[:, 0])
        grid[:, 1] -= np.min(grid[:, 1])
        self.grid = grid

        self.frame1 = tk.Frame(self.root)
        self.frame1.pack(side=tk.LEFT)

        self.frame2 = tk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT)

        self.frame3 = tk.Frame(self.root)
        self.frame3.pack(side=tk.RIGHT)
        #
        # self.frame4 = tk.Frame(self.root)
        # self.frame4.pack(side=tk.RIGHT)

        self.scale_base = self.draw_scale()

        self.canvas = tk.Canvas(self.frame1, width=self.h + 100, height=self.w + 100)
        self.canvas.grid(row=0, column=0)
        # self.canvas.create_text(100, self.h- 100, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))

        self.canvas2 = tk.Canvas(self.frame3, width=self.h + 50, height=self.w + 50)
        self.canvas2.grid(row=1, column=0)


        # fig, axs = plt.subplots(1, figsize=(9, 6))
        # fig.tight_layout(pad=3)
        # self.ax = axs
        # self.canvas3 = FigureCanvasTkAgg(fig, master=self.frame4)
        # self.canvas3.get_tk_widget().grid(row=1, column=1)

        def connection_update_callback():
            self.reset_connections()

        self.pause = False

        def pause_callback():
            self.pause = not self.pause

        self.button = tk.Button(self.frame2, text="Update connection visualization", command=connection_update_callback)
        self.button.grid(row=0, column=1)
        self.set_connections()
        # self.max_connectivity = np.product(self.DDN.connectivity.shape)
        self.button_p = tk.Button(self.frame2, text="Pause/resume", command=pause_callback)
        self.button_p.grid(row=1, column=1)

    def update_plots(self):
        # self.ax[0].clear()
        # self.ax[0].plot(self.time_axis_plots_ms, self.conn_hist)
        # self.ax[0].set_ylim(0, 2)
        # self.ax[0].set_title('Spectral radius')
        # self.ax[0].set_xlabel('Time (ms)')
        # self.ax[0].set_ylabel(r'$\rho(W)$')



        weights = sum(self.DDN.W_masked_list)
        # w_mask = np.array(np.abs(weights) > 1, dtype='float64')
        # w_mask = np.array(weights > 1, dtype='float64')

        weights = np.reshape(weights, weights.shape[0] * weights.shape[1])
        mu_w = round(np.mean(weights), 3)
        std_w = round(np.std(weights), 3)
        inds = np.argwhere(np.abs(weights) > 0)
        weights = weights[inds]


        D = self.DDN.D
        D = np.reshape(D, D.shape[0] * D.shape[1])
        D = D[inds]
        D = D[:, 0]
        weights = weights[:,0]
        # self.ax[0].clear()
        # weights = sum(self.DDN.W_masked_list)
        # weights = np.reshape(weights, weights.shape[0] * weights.shape[1])
        # self.ax[0].hist(weights, bins=list(np.arange(-5, 5, .125)))
        # self.ax[0].set_ylim(0, 3000)
        # self.ax[0].set_xlim(-5, 5)
        # self.ax[0].set_title('Weight distribution. Mean: ' + str(mu_w) + ", Std: " + str(std_w))
        # self.ax[0].set_xlabel('weights')
        # self.ax[0].set_ylabel('frequency')

        # self.ax.clear()
        # self.ax.hist(weights, bins=list(np.arange(-10, 10, .125)))
        # # self.ax.hist(D, bins=list(np.arange(1, 25, 1)))
        # self.ax.hist2d(weights, D, [np.arange(-6, 3, .125), np.arange(1, 14, 1)], density=False, cmax=500)
        # self.ax.set_xlim(-6, 3)
        # # self.ax.set_xlim(-5, 10)
        # self.ax.set_ylim(0, 14)
        # # self.ax.set_ylim(0, 10000)
        #
        # self.ax.set_title('Weight distribution. Mean: ' + str(mu_w) + ", Std: " + str(std_w))
        # self.ax.set_xlabel('weights')
        # self.ax.set_ylabel('Delays')
        # # self.ax[1].clear()
        # self.ax[1].plot(self.time_axis_plots_ms, self.energy_use)
        # self.ax[1].set_ylim(0, 5)
        # self.ax[1].set_title('Network power')
        # self.ax[1].set_xlabel('Time (ms)')
        # self.ax[1].set_ylabel('Power (W)')

        self.canvas3.draw()
        # self.conn_hist[:-1] = self.conn_hist[1:]
        # self.energy_use[:-1] = self.energy_use[1:]
        # spectral_radius = np.max(np.linalg.eigvals(sum(self.DDN.W_masked_list)))
        # self.conn_hist[-1] = spectral_radius
        # self.energy_use[-1] = self.DDN.get_current_power_bio()

    def update_arm(self, target):
        w = self.w
        h = self.h
        s = 50
        arm_base = np.zeros((w, h, 3))
        x, y = self.arm.position()
        xt, yt = target
        p0 = (int(w/2), int(h/2))
        p1 = (p0[0] + int(x[1] * s), p0[1] + int(y[1] * s))
        p2 = (p0[0] + int(x[2] * s), p0[1] + int(y[2] * s))
        t = (p0[0] + int(xt * s), p0[1] + int(yt * s))
        rr, cc = disk((p2[0], p2[1]), self.dot_size)
        arm_base[rr, cc, :] = 1

        rr, cc = disk((t[0], t[1]), self.dot_size)
        arm_base[rr, cc, 0] = 1

        rr, cc, val = line_aa(p0[0], p0[1], p1[0], p1[1])
        arm_base[rr, cc, :] = np.array([val, val, val]).T
        rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1])
        arm_base[rr, cc, :] = np.array([val, val, val]).T
        return arm_base

    def set_connections(self):
        weights = self.DDN.W
        self.connections_pos = None
        self.connections_neg = None
        self.get_connection_base(weights)
        dot_ex, dot_in = self.grid2dots(True)
        all_d = np.stack([dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base], axis=-1)
        connections_all = (self.connections_pos + self.connections_neg)
        self.img_c = np.stack([self.connections_pos, connections_all, self.connections_neg], axis=-1) * \
                     (all_d == 0)
        self.img_c_black = np.stack([connections_all/3, connections_all/3, connections_all/3], axis=-1) * \
                           (all_d == 0)

    def reset_connections(self):
        weights = sum(self.DDN.W_masked_list)
        self.connections_pos = None
        self.connections_neg = None
        self.get_connection_base(weights)
        dot_ex, dot_in = self.grid2dots(True)
        all_d = np.stack([dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base,
                          dot_in + dot_ex + self.scale_base], axis=-1)
        connections_all = self.connections_pos + self.connections_neg
        self.img_c = np.stack([self.connections_pos, connections_all, self.connections_neg], axis=-1) * \
                     (all_d == 0)


    def get_connection_base(self, weights):
        """
        Generates an image of the non-zero connections in this network.
        :return: None
        """

        # weights = np.asarray(weights > 0, dtype='uint8')
        grid = self.grid

        N = grid.shape[0]
        connections_pos = np.ones((self.w, self.h))
        connections_neg = np.ones((self.w, self.h))
        for n1 in range(N):
            for n2 in range(N):
                if weights[n1, n2] != 0:
                    x1 = int(grid[n1, 0] * self.spacing + self.dot_size)
                    y1 = int(grid[n1, 1] * self.spacing + self.dot_size)
                    x2 = int(grid[n2, 0] * self.spacing + self.dot_size)
                    y2 = int(grid[n2, 1] * self.spacing + self.dot_size)
                    connections_pos, connections_neg = draw_line(connections_pos, connections_neg, (x1, y1), (x2, y2),
                                                                 weights[n1, n2])
                    # xa1 = x2 + 0.95 * (x1 - x2) + 5
                    # ya1 = y2 + 0.95 * (y1 - y2) + 5
                    # xa2 = x2 + 0.95 * (x1 - x2) - 5
                    # ya2 = y2 + 0.95 * (y1 - y2) - 5
                    # connections = draw_line(connections, (int(xa1), int(ya1)), (x2, y2))
                    # connections = draw_line(connections, (int(xa2), int(ya2)), (x2, y2))
        self.connections_pos = connections_pos
        self.connections_neg = connections_neg

    def draw_scale(self):
        # compute how many dts fit in a quarter screen
        width = self.h
        dt_width = self.DDN.dt * config.propagation_vel * self.spacing
        n_dt = int(np.floor(.45 * width/dt_width))

        scale_base = np.zeros((self.w, self.h))
        start_pos = np.array([self.w - 30, 20])
        end_pos = start_pos + np.array([0, n_dt * dt_width])
        # main line
        rr, cc, val = line_aa(int(start_pos[0]), int(start_pos[1]), int(end_pos[0]), int(end_pos[1]))
        scale_base[rr, cc] = val
        # vertical markers
        rr, cc, val = line_aa(int(start_pos[0] + 8), int(start_pos[1]), int(start_pos[0] - 8), int(start_pos[1]))
        scale_base[rr, cc] = val
        rr, cc, val = line_aa(int(end_pos[0] + 8), int(end_pos[1]), int(end_pos[0] - 8), int(end_pos[1]))
        scale_base[rr, cc] = val

        for i in range(n_dt):
            rr, cc, val = line_aa(int(start_pos[0] + 5), int(start_pos[1] + i * dt_width),
                                  int(start_pos[0] - 5), int(start_pos[1] + i * dt_width))
            scale_base[rr, cc] = val
        return scale_base


    def grid2dots(self, init=False):
        """
        Generates an image as a ndarray with neuron activation drawn as dots according to coordinates attribute.
        :param init: bool
            Sets activity of all neurons to 1. Only used when initial connection base visualisation is being
            created.
        :return: ndarray, ndarray
            Two w by h array with excitatory and inhibitory neuron activation respectively as dots drawn in the
            array.
        """
        coordinates = self.grid
        act = self.DDN.A[:, 0] * self.DDN.ex_in
        if init:
            act = np.ones_like(act)
        N = coordinates.shape[0]
        dots_ex = np.zeros((self.w, self.h))
        dots_inh = np.zeros((self.w, self.h))
        for i in range(N):
            a = act[i]
            x = np.round(coordinates[i, 0] * self.spacing + self.dot_size)
            y = np.round(coordinates[i, 1] * self.spacing + self.dot_size)
            rr, cc = disk((x, y), self.dot_size)

            if a > 0:
                dots_ex[rr, cc] = a
                dots_inh[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 0
                dots_inh[rr, cc] = -a
        return dots_ex, dots_inh

    def grid2dots_input(self):
        """
        Generates an image as a ndarray with neuron activation drawn as dots according to coordinates attribute.
        :return: ndarray, ndarray
            Two w by h array with excitatory and inhibitory neuron activation respectively as dots drawn in the
            array.
        """
        coordinates = self.grid
        inp_ind = self.DDN.neurons_in
        N = coordinates.shape[0]
        dots_ex = np.zeros((self.w, self.h))
        dots_inh = np.zeros((self.w, self.h))
        dots_inp = np.zeros((self.w, self.h))
        for i in reversed(range(N)):
            # a = act[i]
            x = np.round(coordinates[i, 0] * self.spacing + self.dot_size)
            y = np.round(coordinates[i, 1] * self.spacing + self.dot_size)
            rr, cc = disk((x, y), self.dot_size)
            if i in inp_ind:
                dots_inp[rr, cc] = 1
                dots_ex[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 1
                dots_inp[rr, cc] = 0
        return dots_ex, dots_inp

    def update_a(self, target):
        """
        Updates the gui according to current neuron activity.
        :return: None
        """
        dot_ex, dot_in = self.grid2dots()
        zer = np.zeros_like(self.connections_pos)
        img_d = np.stack([dot_in, dot_ex, zer], axis=-1)
        img = self.img_c + img_d
        img = get_tk_im(img)
        img_arm = self.update_arm(target)
        img_arm = get_tk_im(img_arm)

        while self.pause:
            self.canvas.create_image(5, 5, anchor="nw", image=img)
            self.root.update()

        # self.update_plots()
        self.canvas.create_image(5, 5, anchor="nw", image=img)
        self.canvas2.create_image(10, 40, anchor='nw', image=img_arm)
        self.root.update()

    def get_single_frame(self):
        dot_ex, dot_inp = self.grid2dots_input()
        zer = np.zeros_like(self.connections_pos)
        img_d = np.stack([zer, dot_ex, dot_inp], axis=-1)
        img = self.img_c_black + img_d
        # img = get_tk_im(img)
        return img

    def close(self):
        self.root.destroy()


class EvolutionGui(object):

    def __init__(self, image_list):

        self.dot_size = 5
        self.root = tk.Tk()

        self.frame1 = tk.Frame(self.root)
        self.frame1.pack(side=tk.LEFT)

        self.frame2 = tk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.frame1, width=600, height=600)
        self.canvas.grid(row=0, column=0)

        self.slider = tk.Scale(self.frame2, from_=0, to=len(image_list)-1, length=500)
        self.slider.grid(row=0, column=1)

        self.image_list = [get_tk_im(img) for img in image_list]

    def update(self):
        v = self.slider.get()
        img = self.image_list[v]
        self.canvas.delete('all')
        self.canvas.create_image(5, 5, anchor="nw", image=img)
        self.root.update()


def draw_scale(w, h, dt, spacing):
    # compute how many dts fit in a quarter screen
    width = h
    dt_width = dt * config.propagation_vel * spacing
    n_dt = int(np.floor(.45 * width/dt_width))

    scale_base = np.zeros((w, h))
    start_pos = np.array([w - 30, 20])
    end_pos = start_pos + np.array([0, n_dt * dt_width])
    # main line
    rr, cc, val = line_aa(int(start_pos[0]), int(start_pos[1]), int(end_pos[0]), int(end_pos[1]))
    scale_base[rr, cc] = val
    # vertical markers
    rr, cc, val = line_aa(int(start_pos[0] + 8), int(start_pos[1]), int(start_pos[0] - 8), int(start_pos[1]))
    scale_base[rr, cc] = val
    rr, cc, val = line_aa(int(end_pos[0] + 8), int(end_pos[1]), int(end_pos[0] - 8), int(end_pos[1]))
    scale_base[rr, cc] = val

    for i in range(n_dt):
        rr, cc, val = line_aa(int(start_pos[0] + 5), int(start_pos[1] + i * dt_width),
                              int(start_pos[0] - 5), int(start_pos[1] + i * dt_width))
        scale_base[rr, cc] = val
    return scale_base


def grid2dots_simple(coordinates, w, h, spacing, dot_size=5):

    N = coordinates.shape[0]
    dots_ex = np.zeros((w, h))
    for i in range(N):
        x = np.round(coordinates[i, 0] * spacing + dot_size)
        y = np.round(coordinates[i, 1] * spacing + dot_size)
        rr, cc = disk((x, y), dot_size)

        dots_ex[rr, cc] = 1
    return dots_ex


def grid2dots(coordinates, DDN, w, h, spacing, dot_size=5):

    act = DDN.A[:, 0] * DDN.ex_in
    act = np.ones_like(act) * DDN.ex_in
    N = coordinates.shape[0]
    dots_ex = np.zeros((w, h))
    dots_in = np.zeros((w, h))
    dots_input = np.zeros((w, h))
    for i in range(N):
        a = act[i]
        x = np.round(coordinates[i, 0] * spacing + dot_size)
        y = np.round(coordinates[i, 1] * spacing + dot_size)
        rr, cc = disk((x, y), dot_size)

        if i > DDN.size_in:
            if a > 0:
                dots_ex[rr, cc] = a
                dots_in[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 0
                dots_in[rr, cc] = -a
        else:
            dots_input[rr, cc] = a
    return dots_ex, dots_in, dots_input


def get_connection_base(weights, grid, spacing, w, h, dot_size=5):

    # weights = np.asarray(weights > 0, dtype='uint8')
    N = grid.shape[0]
    connections_pos = np.ones((w, h))
    connections_neg = np.ones((w, h))

    for n1 in range(N):
        for n2 in range(N):
            if weights[n1, n2] != 0:
                x1 = int(grid[n1, 0] * spacing + dot_size)
                y1 = int(grid[n1, 1] * spacing + dot_size)
                x2 = int(grid[n2, 0] * spacing + dot_size)
                y2 = int(grid[n2, 1] * spacing + dot_size)
                connections_pos, connections_neg = draw_line(connections_pos, connections_neg, (x1, y1), (x2, y2),
                                                             weights[n1, n2])
                # xa1 = x2 + 0.95 * (x1 - x2) + 5
                # ya1 = y2 + 0.95 * (y1 - y2) + 5
                # xa2 = x2 + 0.95 * (x1 - x2) - 5
                # ya2 = y2 + 0.95 * (y1 - y2) - 5
                # connections = draw_line(connections, (int(xa1), int(ya1)), (x2, y2))
                # connections = draw_line(connections, (int(xa2), int(ya2)), (x2, y2))
    return connections_pos, connections_neg


def get_network_im(net, dot_size=5):
    grid = net.coordinates
    # shift and scale grid
    grid[:, 0] -= np.min(grid[:, 0])
    grid[:, 1] -= np.min(grid[:, 1])
    width = net.x_range[1] - net.x_range[0]
    height = net.y_range[1] - net.y_range[0]
    ratio = width / height
    if ratio > 1:
        w = 500
        h = int(w / ratio)
    else:
        h = 500
        w = int(h * ratio)

    spacing = w / width
    w += 2*dot_size + 2
    h += 2*dot_size + 2
    connections_pos, connections_neg = get_connection_base(net.W, grid, spacing, w, h, dot_size)
    dots_ex, dots_in, dots_input = grid2dots(grid, net, w, h, spacing, dot_size)
    scale_base = draw_scale(w, h, net.dt, spacing)
    all_d = np.stack([dots_in + dots_ex + dots_input + scale_base,
                      dots_in + dots_ex + scale_base,
                      dots_in + dots_ex + scale_base], axis=-1)
    connections_all = connections_pos + connections_neg

    img_c = np.stack([connections_pos, connections_all, connections_neg], axis=-1) * \
            (all_d == 0)

    zer = np.zeros_like(connections_pos)
    img_d = np.stack([dots_in, dots_ex, zer], axis=-1)
    img = img_c + img_d
    # img = get_tk_im(img)
    return img
