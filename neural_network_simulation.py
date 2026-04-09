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
        
        pareto_mu=1.5,
        
        I_ext_mean=1e-9,
        I_ext_std=5e-10,    # float (A): white noise external input current to each neuron
        
        seed=42,       # int or None: random seed for reproducibility
        record_spikes=True,  # bool: whether to store full spike raster (steps x N)
        record_voltage=True,
        
        weight='gaussian' # gaussian weighted or powerlaw weighted
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
        wI = -4 * w_scale        # inhibitory weight scale (negative, stronger magnitude)

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
            w_min_I = 4 * w_scale * ((self.pareto_mu-1)/self.pareto_mu) # inhibitory minimum magnitude (I stronger)
            
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
            #
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

        return self.spikes, total_spikes

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