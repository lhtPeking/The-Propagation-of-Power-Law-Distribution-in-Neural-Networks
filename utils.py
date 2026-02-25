import os
import numpy as np
from scipy import sparse
import re
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz, csgraph
import matplotlib.pyplot as plt


class ConnectomeAnalysis:
    @staticmethod
    def construct_connection_matrix(csv_path, make_dense=True):
        df = pd.read_csv(csv_path, header=None, names=["pre", "post", "w"])
        df["pre"] = df["pre"].astype(np.int64)
        df["post"] = df["post"].astype(np.int64)
        df["w"] = df["w"].astype(np.float64)
        
        # sum duplicates
        df = df.groupby(["pre", "post"], as_index=False)["w"].sum()
        
        # Mapping: from list to matrix
        node_ids = np.unique(np.concatenate([df["pre"].values, df["post"].values]))
        node_ids.sort()
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        idx_to_id = node_ids
        rows = np.fromiter((id_to_idx[n] for n in df["pre"].values), dtype=np.int64, count=len(df))
        cols = np.fromiter((id_to_idx[n] for n in df["post"].values), dtype=np.int64, count=len(df))
        data = df["w"].values
        A = coo_matrix((data, (rows, cols)), shape=(len(node_ids), len(node_ids))).tocsr() # coo for coordinate list, csr for compressed sparse row
        
        if make_dense:
            A = A.toarray().astype(np.float32, copy=False)

        return A
    
    @staticmethod
    def connection_matrix_scaling(matrix):
        ## -- scale the matrix before compute its eigenspectrum -- ##
        
        
        return scaled_matrix

    @staticmethod
    def compute_eigenspectrum(scaled_matrix, symmetrization=True):
        A = np.asarray(scaled_matrix, dtype=np.float32)
        
        ## -- symmetrization (optional) -- ##
        if symmetrization:
            A = 0.5 * (A + A.T)
            eigenvalue_list = np.linalg.eigvalsh(A) # h for hermitian
        else:
            eigenvalue_list = np.linalg.eigvals(A)
        
        return eigenvalue_list

class PlotMethod:
    @staticmethod
    def log_log_plot(
        data, # list or matrix 
        bin_num, 
        regression=False,
        range_=None, # tuple: (min,max)
        lowerbound=0.025,
        higherbound=0.975,
        num_points=1000, # interception point number
        density=True,
        color_scatter="darkblue",
        color_fit="red",
        show=True,
        ax=None,
        return_params=True,
        xlabel="Log(Connection Strength)",
        ylabel="Log(Probability Density)"
        ):
        
        ## -- data preparation -- ##
        x = np.asarray(data, dtype=float).ravel()
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if range_ is None:
            xmin, xmax = float(np.min(x)), float(np.max(x))
        else:
            xmin, xmax = map(float, range_)
        
        ## -- histogram -- ##
        count, bins = np.histogram(x, bins=bin_num, range=(xmin, xmax), density=density)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mask = (count > 0) & (bin_centers > 0)
        log_count = np.log(count[mask])
        log_bin_centers = np.log(bin_centers[mask])
        
        ## -- scatter --- ##
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.scatter(log_bin_centers, log_count, s=10, color=color_scatter)
        slope = intercept = None
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        
        ## -- regression (optional) -- ##
        if regression:
            n = log_bin_centers.size
            low = int(np.floor(lowerbound * n))
            high = int(np.ceil(higherbound * n))
            x_log_min = float(np.min(log_bin_centers[low:high]))
            x_log_max = float(np.max(log_bin_centers[low:high]))
            uniform_log_bin_centers = np.linspace(x_log_min, x_log_max, int(num_points)) # interception
            uniform_log_count = np.interp(uniform_log_bin_centers, log_bin_centers, log_count)
            slope, intercept = np.polyfit(uniform_log_bin_centers, uniform_log_count, 1) # fit
            fit_line = slope * uniform_log_bin_centers + intercept
            ax.plot(uniform_log_bin_centers, fit_line, linestyle="--", color=color_fit)
            # label=f"Fitted line: slope = {slope:.2f}")
            # ax.legend()
            print(f"Fitted line: slope = {slope:.2f}")

        if show:
            plt.show()

        if return_params:
            return slope, intercept, ax
        
        return ax
        

    
        

class SymmetricActivitySparse:
    """
    symmetric activity-independent model (sparse version)
    only non-zero edges are saved: w_dict[key] = weight, where key = i * N + j (i<j)
    every 'save_every' times update will lead to an automatic save
    """

    def __init__(
        self,
        N,
        s_avg=1.0,
        p=0.5,
        num_updates_per_sample=100,
        burn_factor=50,
        seed=0,
        save_every=None,
        save_dir="checkpoints",
        save_format="edgelist",   # "edgelist" or "npz"
    ):
        self.N = int(N)
        self.E = self.N * (self.N - 1) // 2
        self.p = float(p)
        self.num_updates_per_sample = int(num_updates_per_sample)
        self.burn_updates = int(burn_factor * num_updates_per_sample)
        self.rng = np.random.default_rng(seed)

        # 每次 update 的 prune 比例（近似）
        # 原来是 num_prunes_per_update = ceil(E/num_updates_per_sample)
        # => q ~= 1/num_updates_per_sample
        self.q_prune = 1.0 / self.num_updates_per_sample

        # 初始化总突触数（和你原来一样：s_avg * E）
        self.num_syn = int(round(float(s_avg) * self.E))

        # 稀疏边权字典
        self.w = {}  # key -> weight (int)
        self._init_random_synapses(self.num_syn)

        # autosave
        self.save_every = save_every
        self.save_dir = save_dir
        self.save_format = save_format
        self.iter = 0
        if self.save_every is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            if self.save_format not in ("edgelist", "npz"):
                raise ValueError("save_format must be 'edgelist' or 'npz'")

    # ---------- edge encoding ----------

    def _encode_key(self, i, j):
        # assume i<j
        return i * self.N + j

    def _decode_key(self, key_arr):
        # key = i*N + j
        i = key_arr // self.N
        j = key_arr - i * self.N
        return i.astype(np.int32), j.astype(np.int32)

    def _random_undirected_keys(self, m):
        """
        均匀抽 m 条无向边（允许重复）。
        做法：先均匀抽 directed pair (i, j!=i)，再排序成 undirected。
        """
        i = self.rng.integers(0, self.N, size=m, dtype=np.int64)
        j = self.rng.integers(0, self.N - 1, size=m, dtype=np.int64)
        j = j + (j >= i)  # 保证 j != i

        a = np.minimum(i, j)
        b = np.maximum(i, j)
        return a * self.N + b  # key

    # ---------- init ----------

    def _init_random_synapses(self, num_syn):
        # 随机把 num_syn 个突触丢到随机边上（有放回）
        keys = self._random_undirected_keys(num_syn)
        u, c = np.unique(keys, return_counts=True)
        for key, cnt in zip(u.tolist(), c.tolist()):
            self.w[int(key)] = int(cnt)

    # ---------- saving ----------

    def save(self, tag=None):
        """
        保存当前网络：
          - edgelist: 保存 rows, cols, weights 到 npz（强烈推荐）
          - npz: 保存加权稀疏矩阵 (N,N) 到 .npz（更大更慢）
        """
        if tag is None:
            tag = f"iter_{self.iter:08d}"

        keys = np.fromiter(self.w.keys(), dtype=np.int64, count=len(self.w))
        weights = np.fromiter(self.w.values(), dtype=np.int64, count=len(self.w))

        if self.save_format == "edgelist":
            rows, cols = self._decode_key(keys)
            path = os.path.join(self.save_dir, f"edges_{tag}.npz")
            np.savez_compressed(path, rows=rows, cols=cols, weights=weights, N=self.N)
            return path

        # npz sparse adjacency
        rows, cols = self._decode_key(keys)
        A_upper = sparse.csr_matrix((weights, (rows, cols)), shape=(self.N, self.N))
        A = A_upper + A_upper.T
        path = os.path.join(self.save_dir, f"adj_{tag}.npz")
        sparse.save_npz(path, A)
        return path

    def adjacency_sparse(self, weighted=True):
        """构造当前稀疏邻接矩阵（无向）。"""
        if len(self.w) == 0:
            return sparse.csr_matrix((self.N, self.N), dtype=np.int64 if weighted else np.int8)

        keys = np.fromiter(self.w.keys(), dtype=np.int64, count=len(self.w))
        weights = np.fromiter(self.w.values(), dtype=np.int64, count=len(self.w))
        rows, cols = self._decode_key(keys)

        data = weights.astype(np.int64) if weighted else np.ones_like(weights, dtype=np.int8)
        A_upper = sparse.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        return A_upper + A_upper.T

    # ---------- dynamics ----------

    def one_update(self):
        """
        稀疏近似版 update：
          1) 对当前存在的边，以概率 q_prune prune（近似原来从所有 E 抽一批）
          2) 被移除的突触数 s_temp 再按 Hebb / random 分配
        """
        if len(self.w) == 0:
            # 没边就没法 Hebb；但 random 会新建边（来自后面 s_temp）
            self.iter += 1
            return

        # --- prune existing edges with Bernoulli(q) ---
        keys = np.fromiter(self.w.keys(), dtype=np.int64, count=len(self.w))
        weights = np.fromiter(self.w.values(), dtype=np.int64, count=len(self.w))

        prune_mask = (self.rng.random(keys.shape[0]) < self.q_prune)
        if np.any(prune_mask):
            pruned_keys = keys[prune_mask]
            pruned_weights = weights[prune_mask]
            s_temp = int(pruned_weights.sum())

            # delete pruned edges
            for k in pruned_keys.tolist():
                self.w.pop(int(k), None)
        else:
            s_temp = 0

        if s_temp > 0:
            s_hebb = int(self.rng.binomial(s_temp, self.p))
            s_rand = s_temp - s_hebb

            # --- Hebbian growth: choose edges proportional to current weight ---
            if s_hebb > 0 and len(self.w) > 0:
                keys2 = np.fromiter(self.w.keys(), dtype=np.int64, count=len(self.w))
                weights2 = np.fromiter(self.w.values(), dtype=np.float64, count=len(self.w))
                probs = weights2 / weights2.sum()

                chosen = self.rng.choice(keys2, size=s_hebb, replace=True, p=probs)
                u, c = np.unique(chosen, return_counts=True)
                for k, cnt in zip(u.tolist(), c.tolist()):
                    kk = int(k)
                    self.w[kk] = self.w.get(kk, 0) + int(cnt)
            elif s_hebb > 0 and len(self.w) == 0:
                # 极端情况：全被 prune 了，那 hebb 退化成随机
                s_rand += s_hebb
                s_hebb = 0

            # --- Random growth: uniform random edges ---
            if s_rand > 0:
                rand_keys = self._random_undirected_keys(s_rand)
                u, c = np.unique(rand_keys, return_counts=True)
                for k, cnt in zip(u.tolist(), c.tolist()):
                    kk = int(k)
                    self.w[kk] = self.w.get(kk, 0) + int(cnt)

        # auto save
        self.iter += 1
        if self.save_every is not None and (self.iter % self.save_every == 0):
            self.save()

    def run_updates(self, n_updates):
        for _ in range(int(n_updates)):
            self.one_update()

    def burn_in(self):
        self.run_updates(self.burn_updates)

    # ---------- sampling ----------

    def sample_once(self, compute_clustering=True):
        """
        运行 num_updates_per_sample 次 update 后，返回一次统计：
          vals, counts, density, heterogeneity, clustering
        """
        self.run_updates(self.num_updates_per_sample)

        if len(self.w) == 0:
            return None

        weights = np.fromiter(self.w.values(), dtype=np.int64, count=len(self.w))
        conn_strengths = weights  # 稀疏版里每条非零边的权重就是连接强度

        vals, counts = np.unique(conn_strengths, return_counts=True)

        # density = (#nonzero edges)/E
        density = float(conn_strengths.size / self.E)

        # heterogeneity (你原 MATLAB 同公式)
        p_i = counts / counts.sum()
        diff = np.abs(vals[:, None] - vals[None, :]).astype(float)
        heterogeneity = float(0.5 * np.sum(diff * (p_i[:, None] * p_i[None, :])) / conn_strengths.mean())

        clustering = 0.0
        if compute_clustering:
            A = self.adjacency_sparse(weighted=False)
            clustering = float(clustering_coefficient_undirected_unweighted(A))

        return vals, counts, density, heterogeneity, clustering