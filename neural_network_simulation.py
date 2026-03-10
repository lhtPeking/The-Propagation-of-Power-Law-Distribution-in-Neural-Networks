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