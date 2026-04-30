import numpy as np
import matplotlib.pyplot as plt

class LIFNetwork:
    def __init__(
        self,
        N=1000,          # int: number of neurons in the network
        T=1.0,           # float (s): total simulation time
        dt=1e-3,         # float (s): simulation time step
        tau_m=20e-3,     # float (s): membrane time constant
        V_rest=-65e-3,   # float (V): resting membrane potential
        V_reset=-65e-3,  # float (V): reset potential after a spike
        V_th=-50e-3,     # float (V): spike threshold
        R_m=10e6,        # float (Ohm): membrane resistance
        tau_ref=2e-3,    # float (s): absolute refractory period
        tau_syn=5e-3,    # float (s): synaptic current decay time constant
        
        p_conn=0.1,      # float: connection probability between any two neurons
        w_scale=1e-9,  # float (A): standard deviation (scale) of synaptic weights
        
        g=4.0, # brunel parameter
        
        pareto_mu=1.5,
        
        I_ext_mean=1e-9,
        I_ext_std=5e-10,    # float (A): white noise external input current to each neuron
        
        seed=1,       # int or None: random seed for reproducibility
        record_spikes=True,  # bool: whether to store full spike raster (steps x N)
        record_voltage=True,
        
        weight='gaussian', # gaussian weighted or powerlaw weighted
        W=None
    ):
        
        if seed is not None:
            np.random.seed(seed)
        
        # ===== 1. Simulation parameters =====
        self.N = N                      # number of neurons
        self.T = T                      # total simulation time (s)
        self.dt = dt                    # time step (s)
        self.steps = int(T / dt)        # total number of simulation steps

        # ===== 2. Neuron (LIF) parameters =====
        self.tau_m = tau_m              # membrane time constant (s)
        self.V_rest = V_rest            # resting potential (V)
        self.V_reset = V_reset          # reset potential after spike (V)
        self.V_th = V_th                # spike threshold (V)
        self.R_m = R_m                  # membrane resistance (Ohm)

        self.tau_ref = tau_ref          # absolute refractory period (s)
        self.ref_steps = int(tau_ref / dt)  # refractory period in time steps
        

        # ===== 3. Synaptic parameters =====
        self.tau_syn = tau_syn          # synaptic decay time constant (s)
        # per-step exponential decay factor for synaptic current
        self.alpha = np.exp(-dt / tau_syn)

        self.p_conn = p_conn            # connection probability
        self.w_scale = w_scale          # synaptic weight scale (A)

        self.pareto_mu = pareto_mu

        # Excitatory / inhibitory split
        NE = int(self.N * 0.8)   # number of excitatory neurons (80%)
        NI = self.N - NE         # number of inhibitory neurons (20%)

        # Weight parameters
        wE = w_scale             # excitatory weight scale (positive)
        wI = -g * w_scale        # inhibitory weight scale (negative, stronger magnitude)

        # Connection probabilities
        pE = p_conn              # E → * connection probability
        pI = p_conn              # I → * connection probability

        # Initialize empty weight matrix
        self.W = np.zeros((self.N, self.N), dtype=np.float64)
        
        if weight=='gaussian':
            # ----- 1) Excitatory → all (positive weights) -----
            mask_E = (np.random.rand(NE, self.N) < pE)
            weights_E = np.random.normal(loc=wE, scale=wE * 0.2, size=(NE, self.N))
            self.W[:NE, :] = weights_E * mask_E

            # ----- 2) Inhibitory → all (negative weights) -----
            mask_I = (np.random.rand(NI, self.N) < pI)
            weights_I = np.random.normal(loc=wI, scale=abs(wI) * 0.2, size=(NI, self.N))
            self.W[NE:, :] = weights_I * mask_I
        elif weight=='powerlaw':
            w_min_E = w_scale * ((self.pareto_mu-1)/self.pareto_mu)     # minimum excitatory weight
            w_min_I = g * w_scale * ((self.pareto_mu-1)/self.pareto_mu) # inhibitory minimum magnitude (I stronger)
            
            # ----- 1) Excitatory → all (positive Pareto weights) -----
            mask_E = (np.random.rand(NE, self.N) < pE)

            # pareto(a) returns samples of form (1 + pareto), so multiply by w_min
            weights_E = w_min_E * (1.0 + np.random.pareto(self.pareto_mu, size=(NE, self.N)))

            self.W[:NE, :] = weights_E * mask_E

            # ----- 2) Inhibitory → all (negative Pareto weights) -----
            mask_I = (np.random.rand(NI, self.N) < pI)

            # Generate inhibitory weights: same Pareto distribution but negative
            weights_I = w_min_I * (1.0 + np.random.pareto(self.pareto_mu, size=(NI, self.N)))

            self.W[NE:, :] = -weights_I * mask_I   # negative sign for I → *
        elif weight=='fixed_indegree_gaussian':
            # Brunel-style fixed indegree:
            # for each postsynaptic neuron, sample a fixed number of presynaptic neurons
            # from E and I pools independently.
            # W[src, tgt] = weight from presynaptic src to postsynaptic tgt

            self.W.fill(0.0)

            # fixed indegree for each target neuron
            CE = int(round(pE * NE))   # number of excitatory inputs per target
            CI = int(round(pI * NI))   # number of inhibitory inputs per target

            CE = min(CE, NE)
            CI = min(CI, NI)

            exc_pool = np.arange(NE)           # excitatory source neurons: [0, ..., NE-1]
            inh_pool = np.arange(NE, self.N)   # inhibitory source neurons: [NE, ..., N-1]

            for tgt in range(self.N):
                # 1) fixed number of excitatory presynaptic neurons
                exc_src = np.random.choice(exc_pool, size=CE, replace=False)
                self.W[exc_src, tgt] = np.random.normal(
                    loc=wE,
                    scale=abs(wE) * 0.2,
                    size=CE
                )

                # 2) fixed number of inhibitory presynaptic neurons
                inh_src = np.random.choice(inh_pool, size=CI, replace=False)
                self.W[inh_src, tgt] = np.random.normal(
                    loc=wI,
                    scale=abs(wI) * 0.2,
                    size=CI
                )
                
        elif weight == 'fixed_indegree_powerlaw':
            # Brunel-style fixed indegree with power-law synaptic weights
            # W[src, tgt] = weight from presynaptic src to postsynaptic tgt

            self.W.fill(0.0)

            # fixed indegree for each target neuron
            CE = int(round(pE * NE))   # number of excitatory inputs per target
            CI = int(round(pI * NI))   # number of inhibitory inputs per target

            CE = min(CE, NE)
            CI = min(CI, NI)

            exc_pool = np.arange(NE)           # excitatory source neurons: [0, ..., NE-1]
            inh_pool = np.arange(NE, self.N)   # inhibitory source neurons: [NE, ..., N-1]

            # keep the same mean scale as in the dense powerlaw case
            w_min_E = w_scale * ((self.pareto_mu - 1) / self.pareto_mu)
            w_min_I = g * w_scale * ((self.pareto_mu - 1) / self.pareto_mu)

            for tgt in range(self.N):
                # 1) fixed number of excitatory presynaptic neurons
                exc_src = np.random.choice(exc_pool, size=CE, replace=False)
                exc_weights = w_min_E * (
                    1.0 + np.random.pareto(self.pareto_mu, size=CE)
                )
                self.W[exc_src, tgt] = exc_weights

                # 2) fixed number of inhibitory presynaptic neurons
                inh_src = np.random.choice(inh_pool, size=CI, replace=False)
                inh_weights = w_min_I * (
                    1.0 + np.random.pareto(self.pareto_mu, size=CI)
                )
                self.W[inh_src, tgt] = -inh_weights   # inhibitory weights are negative
            
        elif weight == 'externally_assigned':
            self.W = w_scale * np.array(W, copy=True)

            n_inh = self.N // 2
            inh_neurons = np.random.choice(self.N, size=n_inh, replace=False)
            
            self.W = np.abs(self.W)
            self.W[inh_neurons, :] *= -1



        # ----- 3) No self-connections -----
        np.fill_diagonal(self.W, 0.0)
        

        # ===== 4. External input =====
        self.I_ext_mean = I_ext_mean
        self.I_ext_std = I_ext_std                                    # external constant current (A)
        # self.I_ext_vec = np.full(N, I_ext, dtype=np.float64)    # external current for each neuron

        # ===== 5. State variables =====
        self.record_spikes = record_spikes  # whether to store spike raster
        self.record_voltage = record_voltage
        self.reset_state()                  # initialize dynamic state
        
    def reset_state(self):
        """
        Reset dynamic state variables (V, I_syn, refractory counters, spike buffer).
        Call this before re-running a simulation.
        """
        # membrane potential vector (N,)
        self.V = np.random.uniform(self.V_rest, self.V_th, size=self.N)

        # synaptic current vector (N,)
        self.I_syn = np.zeros(self.N, dtype=np.float64)

        # refractory counter (in time steps) for each neuron (N,)
        # if refr_cnt[i] > 0, neuron i is in refractory period
        self.refr_cnt = np.zeros(self.N, dtype=np.int32)

        # spike raster: shape (steps, N), spikes[t, i] = True if neuron i fires at step t
        if self.record_spikes:
            self.spikes = np.zeros((self.steps, self.N), dtype=bool)
        else:
            self.spikes = None
            
        # Voltage history: shape (steps, N)
        if self.record_voltage:
            self.V_hist = np.zeros((self.steps, self.N), dtype=np.float64)
        else:
            self.V_hist = None

        # index of current time step in simulation loop
        self.current_step = 0
        
    def step(self):
        """
        Advance the network by one time step dt.

        Returns:
            fired : np.ndarray of bool, shape (N,)
                Boolean mask indicating which neurons fired at this time step.
        """
        t_idx = self.current_step
        if t_idx >= self.steps:
            raise RuntimeError("Simulation has already reached the end (no more steps).")

        # ---- 1) Update refractory counters ----
        refractory = (self.refr_cnt > 0)      # neurons currently in refractory state
        self.refr_cnt[refractory] -= 1        # decrement refractory counters
        not_ref = ~refractory                 # neurons allowed to update membrane potential

        # ---- 2) Exponential decay of synaptic current ----
        self.I_syn *= self.alpha              # I_syn(t+dt) = alpha * I_syn(t)

        # ---- 3) Membrane potential update (Euler method) ----
        # Total input current = synaptic current + external guassian current
        I_total = self.I_syn + self.I_ext_mean + self.I_ext_std * np.random.normal(size=self.N)

        # dV/dt = (V_rest - V + R_m * I_total) / tau_m
        dV = self.dt / self.tau_m * (self.V_rest - self.V + self.R_m * I_total)

        # Only update neurons that are not in refractory period
        self.V[not_ref] += dV[not_ref]

        # ---- 4) Threshold detection ----
        fired = (self.V >= self.V_th)         # neurons that crossed threshold
        if self.record_spikes:
            self.spikes[t_idx, :] = fired

        # ---- 5) Reset and set refractory period for fired neurons ----
        self.V[fired] = self.V_reset          # reset membrane potential after spike
        self.refr_cnt[fired] = self.ref_steps # start refractory for these neurons

        # ---- 6) Propagate spikes through synapses ----
        if np.any(fired):
            # spike vector: 1.0 for fired neurons, 0.0 otherwise
            s = fired.astype(np.float64)      # shape (N,)
            # synaptic current increment: I_syn += W @ s
            # note: W[i, j] is weight from presynaptic i to postsynaptic j
            self.I_syn += self.W.T @ s

        # ---- 7) Record voltage ----
        if self.record_voltage:
            self.V_hist[t_idx, :] = self.V

        # advance time step counter
        self.current_step += 1
        return fired
    
    def run(self, T=None, verbose=True):
        """
        Run the simulation.

        Args:
            T : float or None
                If not None, override the total simulation time with T (s).
                The number of steps will be min(int(T / dt), self.steps).
            verbose : bool
                If True, print progress and spike count every 100 steps.

        Returns:
            spikes : np.ndarray or None
                If record_spikes == True:
                    Boolean array of shape (steps_run, N) with spike raster.
                Otherwise:
                    None.
        """
        if T is not None:
            steps = min(int(T / self.dt), self.steps)
        else:
            steps = self.steps

        for k in range(steps):
            fired = self.step()
            if (k + 1) % 100 == 0:
                if self.record_spikes:
                    total_spikes = int(self.spikes[:k+1].sum())
                else:
                    total_spikes = np.nan
                print(f"step {k+1}/{steps}, cumulative spikes = {total_spikes}")

        return self.spikes, total_spikes, self.V_hist

    def get_firing_rates(self):
        """
        Compute average firing rates (Hz) of all neurons based on recorded spikes.

        Returns:
            rates : np.ndarray, shape (N,)
                Average firing rate of each neuron in Hz.
        """
        if self.spikes is None:
            raise RuntimeError("record_spikes is False; no spike raster to compute rates from.")

        # effective simulated time = number of completed steps * dt
        T_sim = self.current_step * self.dt

        # spike count per neuron
        counts = self.spikes[:self.current_step].sum(axis=0)  # shape (N,)
        # firing rate in Hz = spikes per second
        return counts / T_sim
    
    def plot_raster(self, max_neurons=200, figsize=(10, 6), markersize=2):
        """
        Plot a raster plot of recorded spikes.

        Args:
            max_neurons : int
                Maximum number of neurons to show (default 200).
                If N is large, drawing all neurons makes raster unreadable.
            figsize : tuple
                Size of the matplotlib figure.
            markersize : float
                Size of spike dots.
        """

        if not self.record_spikes:
            raise RuntimeError("record_spikes=False, cannot plot raster.")
        if self.current_step == 0:
            raise RuntimeError("Simulation has not been run yet.")

        # Use only the first M neurons (for readability)
        M = min(max_neurons, self.N)

        spike_matrix = self.spikes[:self.current_step, :M]   # shape: (time, M)
        times = np.arange(self.current_step) * self.dt       # x-axis time in seconds

        plt.figure(figsize=figsize, dpi=500)

        # For each neuron i, plot a dot at times where spike_matrix[:, i] == True
        for i in range(M):
            t_spike = times[spike_matrix[:, i]]
            plt.plot(t_spike, np.full_like(t_spike, i),
                    '.', markersize=markersize, color="black")

        plt.xlabel("Time (s)")
        plt.ylabel("Neuron index")
        plt.title(f"Raster Plot (showing {M}/{self.N} neurons)")
        plt.ylim([-1, M])
        plt.tight_layout()
        plt.show()
        
    def plot_voltage(self, max_neurons=20, figsize=(12, 6)):
        """
        Plot membrane voltage traces for a subset of neurons.

        Args:
            max_neurons : int
                Number of neurons to plot (default 20).
            figsize : tuple
                Figure size (width, height).
        """
        if not self.record_voltage:
            raise RuntimeError("record_voltage=False, no voltage history to plot.")

        if self.current_step == 0:
            raise RuntimeError("Simulation has not been run yet.")

        # Select neurons to plot
        M = min(max_neurons, self.N)

        # Voltage history is shape (steps, N)
        Vmat = self.V_hist[:self.current_step, :M]
        times = np.arange(self.current_step) * self.dt

        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize, dpi=500)

        # Plot each neuron’s membrane potential
        for i in range(M):
            plt.plot(times, Vmat[:, i], label=f"Neuron {i}", linewidth=1)

        plt.xlabel("Time (s)")
        plt.ylabel("Membrane potential (V)")
        plt.title(f"Voltage traces (showing {M}/{self.N} neurons)")
        plt.tight_layout()
        plt.show()

    def analyze_covariance(self, max_modes=None, loglog_cov=True, figsize=(10, 10)):
        """
        Compute covariance matrix from voltage traces, plot:
        (1) Eigenvalue spectrum (log-log)
        (2) Covariance distribution (histogram)

        Args:
            max_modes : int or None
                If set, only the largest `max_modes` eigenvalues are plotted.
            loglog_cov : bool
                If True, use log scale for covariance histogram y-axis.
            figsize : tuple
                Size of the entire figure.
        """

        if not self.record_voltage:
            raise RuntimeError("record_voltage=False, voltage history unavailable.")

        if self.current_step == 0:
            raise RuntimeError("Simulation has not been run yet.")

        import matplotlib.pyplot as plt

        # ===== 1) Prepare voltage data =====
        V = self.V_hist[:self.current_step].T    # (N, T)
        V_centered = V - V.mean(axis=1, keepdims=True)

        # ===== 2) Covariance matrix =====
        C = np.cov(V_centered)                   # (N, N)

        # ===== 3) Eigenvalue spectrum =====
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)[::-1]         # descending order

        if max_modes is not None:
            eigvals = eigvals[:max_modes]

        # ===== 4) Covariance distribution (off-diagonal only) =====
        offdiag = C[~np.eye(C.shape[0], dtype=bool)]

        # ===== 5) Plotting =====
        fig, ax = plt.subplots(3, 1, figsize=figsize, dpi=500)

        # ---- (A) Eigen-spectrum (index based)) ----
        ax[0].loglog(eigvals, marker='o', markersize=4, linestyle='-')
        ax[0].set_xlabel("Eigenvalue index")
        ax[0].set_ylabel("Eigenvalue magnitude")
        ax[0].set_title("Eigen-spectrum of Covariance Matrix (log-log)")
        ax[0].grid(True, which="both", ls="--", alpha=0.4)

        # ---- (B) Eigenvalue distribution (log-binned style) ----
        # use absolute value just in case, though covariance eigenvalues should be >= 0
        abs_eigs = np.abs(eigvals)

        # histogram in linear space, density=True gives PDF estimate
        counts, bin_edges = np.histogram(abs_eigs, bins=100000, density=True)

        # bin centers
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # only keep bins with non-zero density (log(0) is invalid)
        mask = counts > 0

        ax[1].loglog(centers[mask], counts[mask], marker='o', linestyle='none')

        ax[1].set_xlabel("Eigenvalue")
        ax[1].set_ylabel("Probability density")
        ax[1].set_title("Distribution of Eigenvalues (log-log)")
        ax[1].grid(True, which="both", ls="--", alpha=0.4)
        
        # ---- (C) Covariance distribution ----
        pos = offdiag[offdiag > 0]
        neg = offdiag[offdiag < 0]
        neg_abs = -neg
        all_abs = np.concatenate([pos, neg_abs])
        
        n_bins = 10000
        bins = np.linspace(all_abs.min(), all_abs.max(), n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        
        def plot_hist_line(data, label, **plot_kwargs):
            if data.size == 0:
                return
            counts, _ = np.histogram(data, bins=bins, density=True)
            mask = counts > 0
            if loglog_cov:
                ax[2].loglog(
                    centers[mask], counts[mask],
                    marker='o', linestyle='none',
                    label=label, **plot_kwargs
                )
            else:
                ax[2].plot(
                    centers[mask], counts[mask],
                    marker='o', linestyle='none',
                    label=label, **plot_kwargs
                )
        
        plot_hist_line(all_abs, label=r"$|C_{ij}|$", color="grey", alpha=0.3)
        plot_hist_line(pos, label=r"$C_{ij} > 0$", color="#A3D78A", alpha=0.3)
        plot_hist_line(neg_abs, label=r"$|C_{ij}|,\ C_{ij}<0$", color="#FF5555", alpha=0.3)
        
        ax[2].set_xlabel("Covariance value")
        ax[2].set_ylabel(r"$P(C_{ij})$")
        ax[2].set_title("Distribution of Off-Diagonal Covariances")
        ax[2].legend()

        plt.tight_layout()
        plt.show()

        return C, eigvals, offdiag
    
    

# ============================================================
# LSM block: fixed recurrent LIF reservoir + trainable readout
# ============================================================    
class LiquidStateMachine(LIFNetwork):
    """
    Liquid State Machine built on top of the existing LIFNetwork.

    Reservoir / liquid:
        - same recurrent spiking LIF network as LIFNetwork
        - recurrent weights are fixed after initialization

    Input:
        - u[t] is projected to reservoir neurons through a fixed sparse W_in
        - alternatively, pass current directly with shape (steps, N)

    Readout:
        - ridge-regression linear readout trained on reservoir state features
        - supports classification labels or continuous targets
    """

    def __init__(
        self,
        n_inputs=1, # input dimension
        input_conn=0.2, # input connects to a certain portion of the network
        input_scale=1.0, # current strength relative to intrinsic connection
        readout_reg=1e-6,
        input_seed=42,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_inputs = int(n_inputs)
        self.input_conn = input_conn
        self.input_scale = input_scale
        self.readout_reg = readout_reg

        # input weights are sampled from connection matrix
        rng = np.random.default_rng(input_seed)
        mask = rng.random((self.n_inputs, self.N)) < input_conn
        candidate_weights = self.W[self.W > 0]
        sampled_weights = rng.choice(
            candidate_weights,
            size=(self.n_inputs, self.N),
            replace=True
        )
        self.W_in = input_scale * sampled_weights * mask

        # Readout parameters are created after fit_readout()
        self.readout_W = None
        self.readout_classes_ = None
        self.readout_is_classifier_ = False

    def _format_input(self, u):
        """
        Convert input into a 2D array.

        Accepted shapes:
            None              -> zeros, shape (steps, n_inputs)
            (steps,)          -> scalar time series, shape (steps, 1)
            (steps, n_inputs) -> projected by W_in
        """
        if u is None:
            return np.zeros((self.steps, self.n_inputs), dtype=np.float64)

        u = np.asarray(u, dtype=np.float64)

        if u.ndim == 1:
            u = u[:, None]

        if u.ndim != 2:
            raise ValueError("Input u must have shape (steps,), (steps, n_inputs), or (steps, N).")

        if u.shape[1] != self.n_inputs:
            raise ValueError(
                f"Input second dimension must be n_inputs={self.n_inputs}, "
                f"got {u.shape[1]}."
            )

        if u.shape[0] > self.steps:
            u = u[:self.steps]
        elif u.shape[0] < self.steps:
            pad = np.zeros((self.steps - u.shape[0], u.shape[1]), dtype=np.float64)
            u = np.vstack([u, pad])

        return u

    def _input_to_current(self, u_t):
        """
        Map one input sample u_t into reservoir input current.
        """
        u_t = np.asarray(u_t, dtype=np.float64)

        if u_t.shape != (self.n_inputs,):
            raise ValueError(
                f"u_t must have shape ({self.n_inputs},), got {u_t.shape}."
            )

        return u_t @ self.W_in

    def step_with_input(self, I_input=None):
        """
        One LIF update step with optional LSM input current.

        I_input should be shape (N,), in Ampere.
        """
        t_idx = self.current_step
        if t_idx >= self.steps:
            raise RuntimeError("Simulation has already reached the end (no more steps).")

        if I_input is None:
            I_input = 0.0
        else:
            I_input = np.asarray(I_input, dtype=np.float64)
            if I_input.shape != (self.N,):
                raise ValueError(f"I_input must have shape ({self.N},), got {I_input.shape}.")

        refractory = self.refr_cnt > 0
        self.refr_cnt[refractory] -= 1
        not_ref = ~refractory

        self.I_syn *= self.alpha

        I_noise = self.I_ext_std * np.random.normal(size=self.N)
        I_total = self.I_syn + self.I_ext_mean + I_noise + I_input

        dV = self.dt / self.tau_m * (self.V_rest - self.V + self.R_m * I_total)
        self.V[not_ref] += dV[not_ref]

        fired = self.V >= self.V_th

        if self.record_spikes:
            self.spikes[t_idx, :] = fired

        self.V[fired] = self.V_reset
        self.refr_cnt[fired] = self.ref_steps

        if np.any(fired):
            s = fired.astype(np.float64)
            self.I_syn += self.W.T @ s

        if self.record_voltage:
            self.V_hist[t_idx, :] = self.V

        self.current_step += 1
        return fired

    def run_lsm_trial(
        self,
        u=None,
        reset=True,
        feature="filtered_spikes",
        filter_tau=30e-3,
        readout_window=None,
        return_state_trace=False
    ):
        """
        Run one input trial through the liquid and return a readout feature vector.

        Args:
            u : array or None
                Input time series. See _format_input().
            reset : bool
                Reset reservoir state before this trial.
            feature : {'filtered_spikes', 'spike_counts', 'voltage_mean', 'last_voltage'}
                Reservoir state representation for readout.
            filter_tau : float
                Time constant for low-pass filtered spike state.
            readout_window : tuple, int, or None
                Which time steps to use for readout.
                - None: use the whole trial
                - int: use the last `readout_window` steps
                - (start, stop): use [start:stop]
            return_state_trace : bool
                If True, also return the full filtered state trace.

        Returns:
            features : ndarray, shape (N,)
            state_trace : optional ndarray, shape (steps, N)
        """
        if reset:
            self.reset_state()

        u = self._format_input(u)

        beta = np.exp(-self.dt / filter_tau)
        x = np.zeros(self.N, dtype=np.float64)
        state_trace = np.zeros((self.steps, self.N), dtype=np.float64)

        spike_counts = np.zeros(self.N, dtype=np.float64)

        # Key Iteration Loop
        for t in range(self.steps):
            I_input = self._input_to_current(u[t])
            fired = self.step_with_input(I_input)

            s = fired.astype(np.float64)
            spike_counts += s
            x = beta * x + s
            state_trace[t, :] = x

        if readout_window is None:
            sl = slice(0, self.steps)
        elif isinstance(readout_window, int):
            sl = slice(max(0, self.steps - readout_window), self.steps)
        else:
            start, stop = readout_window
            sl = slice(start, stop)

        if feature == "filtered_spikes":
            features = state_trace[sl].mean(axis=0)
        elif feature == "spike_counts":
            steps_in_window = max(1, state_trace[sl].shape[0])
            if readout_window is None:
                features = spike_counts / (self.steps * self.dt)
            else:
                # recompute counts from recorded spikes for the selected window
                features = self.spikes[sl].sum(axis=0) / (steps_in_window * self.dt)
        elif feature == "voltage_mean":
            if self.V_hist is None:
                raise RuntimeError("record_voltage=False; cannot use voltage_mean features.")
            features = self.V_hist[sl].mean(axis=0)
        elif feature == "last_voltage":
            if self.V_hist is None:
                raise RuntimeError("record_voltage=False; cannot use last_voltage features.")
            features = self.V_hist[self.current_step - 1].copy()
        else:
            raise ValueError("feature must be 'filtered_spikes', 'spike_counts', 'voltage_mean', or 'last_voltage'.")

        if return_state_trace:
            return features, state_trace
        return features

    def build_lsm_dataset(self, input_trials, labels=None, **trial_kwargs):
        """
        Convert a list/array of input trials into readout features X.
        """
        X = []
        for u in input_trials:
            X.append(self.run_lsm_trial(u=u, reset=True, **trial_kwargs))
        X = np.asarray(X, dtype=np.float64) # X.shape = (n_trials, N)

        if labels is None:
            return X
        return X, np.asarray(labels)

    def fit_readout(self, X, y, reg=None, add_bias=True, classifier=True):
        """
        Train ridge-regression readout.

        For classification, y should be class labels. The method one-hot encodes y
        and prediction uses argmax.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_trials, n_features).")

        if reg is None:
            reg = self.readout_reg

        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.X_mean_) / self.X_std_

        if add_bias:
            X_aug = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])
        else:
            X_aug = X_norm

        self.readout_is_classifier_ = bool(classifier)

        if classifier:
            classes = np.unique(y)
            self.readout_classes_ = classes
            # one-hot
            Y = np.zeros((len(y), len(classes)), dtype=np.float64)
            for k, c in enumerate(classes):
                Y[y == c, k] = 1.0
        else:
            self.readout_classes_ = None
            Y = y.astype(np.float64)
            if Y.ndim == 1:
                Y = Y[:, None]
                
        penalty = np.eye(X_aug.shape[1])
        if add_bias:
            penalty[-1, -1] = 0.0

        A = X_aug.T @ X_aug + reg * penalty
        B = X_aug.T @ Y

        self.readout_W = np.linalg.solve(A, B)
        self.readout_add_bias_ = add_bias
        return self

    def predict_readout(self, X):
        if self.readout_W is None:
            raise RuntimeError("Readout has not been trained. Call fit_readout() first.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X[None, :]

        X = (X - self.X_mean_) / self.X_std_

        if self.readout_add_bias_:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        scores = X @ self.readout_W

        if self.readout_is_classifier_:
            idx = np.argmax(scores, axis=1)
            return self.readout_classes_[idx]

        if scores.shape[1] == 1:
            return scores[:, 0]
        return scores

    def fit_lsm(self, input_trials, labels, **trial_kwargs):
        """
        Convenience wrapper: run trials, build features, and fit readout.
        """
        X, y = self.build_lsm_dataset(input_trials, labels, **trial_kwargs)
        self.fit_readout(X, y)
        return X

    def score_lsm(self, input_trials, labels, **trial_kwargs):
        """
        Classification accuracy on a set of trials.
        """
        X, y = self.build_lsm_dataset(input_trials, labels, **trial_kwargs)
        pred = self.predict_readout(X)
        return np.mean(pred == y)


    def make_binary_spike_train(self, rate_hz=20.0, seed=None):
        """
        Generate one binary input spike train.

        Return:
            u: shape (steps,)
        """
        rng = np.random.default_rng(seed)
        p = rate_hz * self.dt
        return (rng.random(self.steps) < p).astype(np.float64)


    def perturb_spike_train_by_distance(self, u, distance=0.1, seed=None):
        """
        Create v from u by flipping approximately `distance` fraction of time bins.

        distance = 0.0 -> identical input
        distance = 0.1 -> 10% bins changed
        distance = 0.4 -> 40% bins changed
        """
        rng = np.random.default_rng(seed)

        u = np.asarray(u, dtype=np.float64).copy()
        if u.ndim != 1:
            raise ValueError("u must have shape (steps,).")

        v = u.copy()
        n_flip = int(round(distance * self.steps))

        flip_idx = rng.choice(self.steps, size=n_flip, replace=False)
        v[flip_idx] = 1.0 - v[flip_idx]

        return v


    def compute_state_distance_trace(
        self,
        u,
        v,
        filter_tau=30e-3
    ):
        """
        Run two inputs u and v in separate trials and compute
        ||x_u(t) - x_v(t)|| over time.

        Return:
            dist_trace: shape (steps,)
        """
        _, state_u = self.run_lsm_trial(
            u=u,
            reset=True,
            feature="filtered_spikes",
            filter_tau=filter_tau,
            return_state_trace=True
        )

        _, state_v = self.run_lsm_trial(
            u=v,
            reset=True,
            feature="filtered_spikes",
            filter_tau=filter_tau,
            return_state_trace=True
        )

        dist_trace = np.linalg.norm(state_u - state_v, axis=1)

        return dist_trace


    def run_separation_experiment(
        self,
        distances=(0.0, 0.1, 0.2, 0.4),
        n_pairs=200,
        rate_hz=20.0,
        filter_tau=30e-3,
        seed=0
    ):
        """
        Maass-style separation experiment.

        For each input distance d:
            1. Generate random spike train u.
            2. Create v by perturbing u with distance d.
            3. Run u and v separately through the liquid.
            4. Compute ||x_u(t) - x_v(t)||.
            5. Average over n_pairs.

        Return:
            results: dict
                results[d] = average distance trace, shape (steps,)
        """
        rng = np.random.default_rng(seed)

        results = {}

        for d in distances:
            traces = []

            for _ in range(n_pairs):
                u = self.make_binary_spike_train(
                    rate_hz=rate_hz,
                    seed=rng.integers(1_000_000_000)
                )

                v = self.perturb_spike_train_by_distance(
                    u,
                    distance=d,
                    seed=rng.integers(1_000_000_000)
                )

                dist_trace = self.compute_state_distance_trace(
                    u=u,
                    v=v,
                    filter_tau=filter_tau
                )

                traces.append(dist_trace)

            results[d] = np.mean(np.asarray(traces), axis=0)

        return results


    def plot_separation_experiment(
        self,
        results,
        figsize=(8, 5)
    ):
        """
        Plot average liquid-state distance over time.
        """
        import matplotlib.pyplot as plt

        times = np.arange(self.steps) * self.dt

        plt.figure(figsize=figsize, dpi=200)

        for d, dist_trace in results.items():
            plt.plot(times, dist_trace, label=f"d(u,v)={d}")

        plt.xlabel("time [sec]")
        plt.ylabel("state distance")
        plt.title("Average distance of liquid states")
        plt.legend()
        plt.tight_layout()
        plt.show()



# ============================================================
# Local-stimulation propagation experiment
# ============================================================
class StimulusSpreadNetwork(LIFNetwork):
    """
    LIFNetwork subclass for local-stimulation propagation experiments.

    This class is intentionally separate from LiquidStateMachine:
    - LiquidStateMachine is a reservoir-computing/readout wrapper.
    - StimulusSpreadNetwork is a pure network-dynamics experiment.

    Experiment:
        1. Randomly choose n_stim neurons.
        2. Force them to spike at the refractory-limited maximum rate
           during a short stimulation window.
        3. Measure firing of all non-stimulated neurons during the following
           observation window, default 100 ms.
    """

    def step_with_forced_spikes(self, I_input=None, forced_spike_mask=None, respect_refractory=True):
        """
        One simulation step, with optional externally forced spikes.

        Args:
            I_input : None or ndarray, shape (N,)
                Extra input current in Ampere.
            forced_spike_mask : None or ndarray[bool], shape (N,)
                Neurons forced to fire at this step.
            respect_refractory : bool
                If True, forced neurons can only fire when not refractory.
                This makes the maximum forced rate approximately 1 / tau_ref.
                If False, forced neurons can fire every dt, which is usually
                less biologically reasonable.

        Returns:
            fired : ndarray[bool], shape (N,)
        """
        t_idx = self.current_step
        if t_idx >= self.steps:
            raise RuntimeError("Simulation has already reached the end (no more steps).")

        if I_input is None:
            I_input = 0.0
        else:
            I_input = np.asarray(I_input, dtype=np.float64)
            if I_input.shape != (self.N,):
                raise ValueError(f"I_input must have shape ({self.N},), got {I_input.shape}.")

        if forced_spike_mask is None:
            forced_spike_mask = np.zeros(self.N, dtype=bool)
        else:
            forced_spike_mask = np.asarray(forced_spike_mask, dtype=bool)
            if forced_spike_mask.shape != (self.N,):
                raise ValueError(
                    f"forced_spike_mask must have shape ({self.N},), got {forced_spike_mask.shape}."
                )

        # 1) refractory update
        refractory = self.refr_cnt > 0
        self.refr_cnt[refractory] -= 1
        not_ref = ~refractory

        # 2) synaptic current decay
        self.I_syn *= self.alpha

        # 3) membrane update
        I_noise = self.I_ext_std * np.random.normal(size=self.N)
        I_total = self.I_syn + self.I_ext_mean + I_noise + I_input

        dV = self.dt / self.tau_m * (self.V_rest - self.V + self.R_m * I_total)
        self.V[not_ref] += dV[not_ref]

        # 4) natural threshold spikes plus externally forced spikes
        fired = self.V >= self.V_th
        if respect_refractory:
            fired = fired | (forced_spike_mask & not_ref)
        else:
            fired = fired | forced_spike_mask

        # 5) record, reset, and set refractory period
        if self.record_spikes:
            self.spikes[t_idx, :] = fired

        self.V[fired] = self.V_reset
        self.refr_cnt[fired] = self.ref_steps

        # 6) recurrent propagation
        if np.any(fired):
            self.I_syn += self.W.T @ fired.astype(np.float64)

        # 7) voltage recording
        if self.record_voltage:
            self.V_hist[t_idx, :] = self.V

        self.current_step += 1
        return fired

    def run_stimulus_spread_experiment(
        self,
        n_stim,
        stim_duration=20e-3,
        observe_duration=100e-3,
        baseline_duration=50e-3,
        target_pool="excitatory",
        stimulated_neurons=None,
        reset=True,
        seed=None,
        respect_refractory=True,
        verbose=True,
    ):
        """
        Randomly stimulate n_stim neurons and measure whether activity spreads.

        Args:
            n_stim : int
                Number of neurons to stimulate.
            stim_duration : float
                Duration of forced stimulation, in seconds.
            observe_duration : float
                Post-stimulus observation window, in seconds. Default 100 ms.
            baseline_duration : float
                Pre-stimulus baseline period, in seconds.
            target_pool : {'excitatory', 'inhibitory', 'all'}
                Pool from which stimulated neurons are sampled.
            stimulated_neurons : None or array-like[int]
                Explicit neuron indices to stimulate. If provided, this overrides
                n_stim and target_pool.
            reset : bool
                Whether to reset network state before running the experiment.
            seed : int or None
                Random seed for choosing stimulated neurons.
            respect_refractory : bool
                If True, maximum forced rate is refractory-limited.
            verbose : bool
                Print summary statistics.

        Returns:
            result : dict
                Contains spike matrix, stimulated neuron indices, observed rates,
                participation fraction, and timing metadata.
        """
        if not self.record_spikes:
            raise RuntimeError("record_spikes must be True for this experiment.")

        if reset:
            self.reset_state()

        rng = np.random.default_rng(seed)

        n_stim = int(n_stim)
        if n_stim <= 0 or n_stim >= self.N:
            raise ValueError("n_stim must be between 1 and self.N - 1.")

        NE = int(self.N * 0.8)

        if stimulated_neurons is None:
            if target_pool == "excitatory":
                pool = np.arange(NE)
            elif target_pool == "inhibitory":
                pool = np.arange(NE, self.N)
            elif target_pool == "all":
                pool = np.arange(self.N)
            else:
                raise ValueError("target_pool must be 'excitatory', 'inhibitory', or 'all'.")

            if n_stim > len(pool):
                raise ValueError("n_stim is larger than the selected target pool.")

            stimulated_neurons = rng.choice(pool, size=n_stim, replace=False)
        else:
            stimulated_neurons = np.asarray(stimulated_neurons, dtype=int)
            n_stim = stimulated_neurons.size
            if n_stim <= 0 or n_stim >= self.N:
                raise ValueError("stimulated_neurons must contain between 1 and self.N - 1 neurons.")
            if np.any(stimulated_neurons < 0) or np.any(stimulated_neurons >= self.N):
                raise ValueError("stimulated_neurons contains invalid neuron indices.")

        stim_mask = np.zeros(self.N, dtype=bool)
        stim_mask[stimulated_neurons] = True
        other_mask = ~stim_mask

        baseline_steps = int(round(baseline_duration / self.dt))
        stim_steps = int(round(stim_duration / self.dt))
        observe_steps = int(round(observe_duration / self.dt))
        total_steps = baseline_steps + stim_steps + observe_steps

        if total_steps > self.steps:
            raise ValueError(
                f"Experiment needs {total_steps} steps, but network has only {self.steps}. "
                "Increase T or shorten baseline/stimulus/observation durations."
            )

        # Baseline.
        for _ in range(baseline_steps):
            self.step_with_forced_spikes(None, None, respect_refractory)

        stim_start = self.current_step

        # Stimulation: selected neurons are forced to fire at the maximum
        # allowed by refractory dynamics.
        for _ in range(stim_steps):
            self.step_with_forced_spikes(None, stim_mask, respect_refractory)

        observe_start = self.current_step

        # Observation: no forced spikes. Measure propagation through recurrent dynamics.
        for _ in range(observe_steps):
            self.step_with_forced_spikes(None, None, respect_refractory)

        observe_stop = self.current_step

        spikes = self.spikes[:self.current_step].copy()

        baseline_spikes = spikes[:baseline_steps] if baseline_steps > 0 else np.zeros((0, self.N), dtype=bool)
        stim_spikes = spikes[stim_start:observe_start]
        observe_spikes = spikes[observe_start:observe_stop]

        stimulated_counts = stim_spikes[:, stim_mask].sum(axis=0)
        stimulated_rates = stimulated_counts / max(stim_duration, self.dt)

        other_counts_observe = observe_spikes[:, other_mask].sum(axis=0)
        other_rates_observe = other_counts_observe / max(observe_duration, self.dt)

        if baseline_steps > 0:
            other_baseline_counts = baseline_spikes[:, other_mask].sum(axis=0)
            other_baseline_rates = other_baseline_counts / max(baseline_duration, self.dt)
            baseline_mean_other_rate = float(other_baseline_rates.mean())
        else:
            other_baseline_rates = None
            baseline_mean_other_rate = np.nan

        mean_other_rate = float(other_rates_observe.mean())
        participation_fraction = float(np.mean(other_counts_observe > 0))
        total_other_spikes = int(other_counts_observe.sum())

        result = {
            "n_stim": n_stim,
            "stimulated_neurons": stimulated_neurons,
            "stim_mask": stim_mask,
            "other_mask": other_mask,
            "baseline_steps": baseline_steps,
            "stim_steps": stim_steps,
            "observe_steps": observe_steps,
            "stim_start_step": stim_start,
            "observe_start_step": observe_start,
            "observe_stop_step": observe_stop,
            "stim_duration": stim_duration,
            "observe_duration": observe_duration,
            "baseline_duration": baseline_duration,
            "respect_refractory": respect_refractory,
            "stimulated_rates_during_stim": stimulated_rates,
            "other_rates_observe": other_rates_observe,
            "other_baseline_rates": other_baseline_rates,
            "mean_other_rate": mean_other_rate,
            "median_other_rate": float(np.median(other_rates_observe)),
            "baseline_mean_other_rate": baseline_mean_other_rate,
            "participation_fraction": participation_fraction,
            "total_other_spikes": total_other_spikes,
            "spikes": spikes,
        }

        if verbose:
            max_rate = 1.0 / self.tau_ref if respect_refractory else 1.0 / self.dt
            print("Stimulus-spread experiment")
            print(f"  stimulated neurons: {n_stim} / {self.N} ({100*n_stim/self.N:.2f}%)")
            print(f"  theoretical forced max rate: {max_rate:.1f} Hz")
            print(f"  stimulated mean rate during stimulus: {stimulated_rates.mean():.2f} Hz")
            print(
                f"  other-neuron mean rate in next {observe_duration*1e3:.0f} ms: "
                f"{mean_other_rate:.2f} Hz"
            )
            print(f"  other-neuron participation fraction: {participation_fraction:.3f}")
            if baseline_steps > 0:
                print(f"  baseline other-neuron mean rate: {baseline_mean_other_rate:.2f} Hz")

        return result

    def sweep_stimulus_spread(
        self,
        n_stim_values,
        repeats=5,
        stim_duration=20e-3,
        observe_duration=100e-3,
        baseline_duration=50e-3,
        target_pool="excitatory",
        seed=0,
        **kwargs,
    ):
        """
        Repeat the stimulus-spread experiment across different n_stim values.

        Returns:
            summary : dict
                Mean/std statistics for each n_stim.
            raw_results : list[dict]
                Individual experiment results.
        """
        rng = np.random.default_rng(seed)
        raw_results = []
        rows = []

        for n_stim in n_stim_values:
            for _ in range(repeats):
                trial_seed = int(rng.integers(0, 2**31 - 1))
                res = self.run_stimulus_spread_experiment(
                    n_stim=n_stim,
                    stim_duration=stim_duration,
                    observe_duration=observe_duration,
                    baseline_duration=baseline_duration,
                    target_pool=target_pool,
                    reset=True,
                    seed=trial_seed,
                    verbose=False,
                    **kwargs,
                )

                raw_results.append(res)
                rows.append(
                    (
                        n_stim,
                        res["mean_other_rate"],
                        res["participation_fraction"],
                        res["total_other_spikes"],
                        res["baseline_mean_other_rate"],
                    )
                )

        rows = np.asarray(rows, dtype=float)
        n_vals = np.asarray(list(n_stim_values), dtype=int)

        summary = {
            "n_stim_values": n_vals,
            "mean_other_rate": np.array([rows[rows[:, 0] == n, 1].mean() for n in n_vals]),
            "std_other_rate": np.array(
                [rows[rows[:, 0] == n, 1].std(ddof=1) if np.sum(rows[:, 0] == n) > 1 else 0.0 for n in n_vals]
            ),
            "mean_participation_fraction": np.array([rows[rows[:, 0] == n, 2].mean() for n in n_vals]),
            "std_participation_fraction": np.array(
                [rows[rows[:, 0] == n, 2].std(ddof=1) if np.sum(rows[:, 0] == n) > 1 else 0.0 for n in n_vals]
            ),
            "mean_total_other_spikes": np.array([rows[rows[:, 0] == n, 3].mean() for n in n_vals]),
            "mean_baseline_other_rate": np.array([np.nanmean(rows[rows[:, 0] == n, 4]) for n in n_vals]),
            "rows": rows,
            "raw_results": raw_results,
        }

        return summary, raw_results

    def plot_stimulus_spread_result(self, result, max_neurons=300, figsize=(10, 7), markersize=2):
        """
        Plot raster and population firing rate for one stimulus-spread result.
        """
        spikes = result["spikes"]
        steps = spikes.shape[0]
        times = np.arange(steps) * self.dt
        M = min(max_neurons, self.N)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=300, sharex=True)

        stim_set = set(result["stimulated_neurons"].tolist())
        for i in range(M):
            t_spike = times[spikes[:, i]]
            if i in stim_set:
                ax[0].plot(t_spike, np.full_like(t_spike, i), ".", markersize=markersize)
            else:
                ax[0].plot(t_spike, np.full_like(t_spike, i), ".", markersize=markersize, color="black")

        stim_start_t = result["stim_start_step"] * self.dt
        observe_start_t = result["observe_start_step"] * self.dt
        observe_stop_t = result["observe_stop_step"] * self.dt

        ax[0].axvspan(stim_start_t, observe_start_t, alpha=0.2, label="forced stimulus")
        ax[0].axvspan(observe_start_t, observe_stop_t, alpha=0.1, label="observation")
        ax[0].set_ylabel("Neuron index")
        ax[0].set_title(f"Stimulus spread raster, n_stim={result['n_stim']}")
        ax[0].legend(loc="upper right")

        bin_steps = max(1, int(round(5e-3 / self.dt)))
        n_bins = steps // bin_steps

        if n_bins > 0:
            pop_rate = (
                spikes[: n_bins * bin_steps]
                .reshape(n_bins, bin_steps, self.N)
                .sum(axis=(1, 2))
                / (bin_steps * self.dt * self.N)
            )
            bin_times = (np.arange(n_bins) + 0.5) * bin_steps * self.dt
            ax[1].plot(bin_times, pop_rate)

        ax[1].axvspan(stim_start_t, observe_start_t, alpha=0.2)
        ax[1].axvspan(observe_start_t, observe_stop_t, alpha=0.1)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Population rate (Hz/neuron)")
        ax[1].set_title("Whole-network population firing rate")

        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_stimulus_sweep(self, summary, figsize=(8, 5)):
        """
        Plot spread statistics versus number of stimulated neurons.
        """
        n = summary["n_stim_values"]

        fig, ax1 = plt.subplots(figsize=figsize, dpi=300)

        ax1.errorbar(
            n,
            summary["mean_other_rate"],
            yerr=summary["std_other_rate"],
            marker="o",
            capsize=3,
        )
        ax1.set_xlabel("Number of stimulated neurons")
        ax1.set_ylabel("Other-neuron mean rate in observation window (Hz)")

        ax2 = ax1.twinx()
        ax2.errorbar(
            n,
            summary["mean_participation_fraction"],
            yerr=summary["std_participation_fraction"],
            marker="s",
            capsize=3,
        )
        ax2.set_ylabel("Participation fraction of other neurons")
        ax2.set_ylim(0, 1.05)

        plt.title("Stimulus spread versus seed size")
        plt.tight_layout()
        plt.show()

        return fig, (ax1, ax2)