import os
import numpy as np
from scipy import sparse
import re
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz, csgraph, issparse
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
        # This step could be ignored if all datasets have a same N
        
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
    
    @staticmethod
    def compute_shuffled_eigenspectrum(scaled_matrix, symmetrization=True, shuffle_diagonal=True, seed=42):
        A = np.asarray(scaled_matrix, dtype=np.float32)
        if symmetrization:
            A = 0.5 * (A + A.T)
            
        n = A.shape[0]
        rng = np.random.default_rng(seed)
        
        iu = np.triu_indices(n, k=1)
        upper_vals = A[iu].copy()
        rng.shuffle(upper_vals)
        
        A_shuf = np.zeros_like(A)
        A_shuf[iu] = upper_vals
        A_shuf[(iu[1], iu[0])] = upper_vals
        
        diag = np.diag(A).copy()
        if shuffle_diagonal:
            rng.shuffle(diag)
            
        np.fill_diagonal(A_shuf, diag)
        
        eigenvalue_list = np.linalg.eigvalsh(A_shuf)

        return eigenvalue_list
    
    @staticmethod
    def compute_brunel_weight_eigenspectrum(scaled_matrix, symmetrization=True, neg_fraction=1/5, scale=4.0, seed=42):
        A = np.asarray(scaled_matrix, dtype=np.float32)
        n = A.shape[0]
        if symmetrization:
            A = 0.5 * (A + A.T)

        rng = np.random.default_rng(seed)
        triu_r, triu_c = np.triu_indices(n, k=0) # k=0 includes diagonal, r for row, c for column
        num_unique = triu_r.size
        k = int(round(neg_fraction * num_unique))
        
        pick = rng.choice(num_unique, size=k, replace=False)
        r = triu_r[pick]
        c = triu_c[pick]

        A[r, c] = -np.abs(A[r, c]) * scale
        A[c, r] = A[r, c]

        if symmetrization:
            eigenvalue_list = np.linalg.eigvalsh(A)
        else:
            eigenvalue_list = np.linalg.eigvals(A)

        return eigenvalue_list

    @staticmethod
    def compute_shuffled_brunel_weight_eigenspectrum(scaled_matrix, symmetrization=True, shuffle_diagonal=True, neg_fraction=1/5, scale=4.0, seed=42):
        A = np.asarray(scaled_matrix, dtype=np.float32)

        if symmetrization:
            A = 0.5 * (A + A.T)

        n = A.shape[0]
        rng = np.random.default_rng(seed)
        
        # Step 1: shuffle upper triangle
        iu = np.triu_indices(n, k=1)
        upper_vals = A[iu].copy()
        rng.shuffle(upper_vals)

        A_shuf = np.zeros_like(A)
        A_shuf[iu] = upper_vals
        A_shuf[(iu[1], iu[0])] = upper_vals

        diag = np.diag(A).copy()
        if shuffle_diagonal:
            rng.shuffle(diag)
        np.fill_diagonal(A_shuf, diag)

        # Step 2: apply Brunel-style weights
        triu_r, triu_c = np.triu_indices(n, k=0)
        num_unique = triu_r.size
        k = int(round(neg_fraction * num_unique))
        pick = rng.choice(num_unique, size=k, replace=False)
        r = triu_r[pick]
        c = triu_c[pick]

        A_shuf[r, c] = -np.abs(A_shuf[r, c]) * scale
        A_shuf[c, r] = A_shuf[r, c]

        # Step 3: eigenspectrum
        if symmetrization:
            eigenvalue_list = np.linalg.eigvalsh(A_shuf)
        else:
            eigenvalue_list = np.linalg.eigvals(A_shuf)

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
        log_count = np.log10(count[mask])
        log_bin_centers = np.log10(bin_centers[mask])
        
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
        
    @staticmethod
    def MLE_regression(
        data,
        xmin_num=100,
        xmin_range=None,          # tuple: (xmin_low, xmin_high)
        min_tail_size=20,         # require at least this many points in the tail
        ax=None,
        show=True,
        color_body="gray",        # x < xmin
        color_tail="darkblue",    # x >= xmin
        color_fit="red",
        color_xmin="red",
        xlabel="x",
        ylabel="Pr(X > x)",
        title="Power-law fit on CCDF",
        return_details=True,
        show_legend=False
    ):
        """
        Continuous power-law fit:
            p(x) ~ x^{-alpha}, for x >= xmin

        CCDF:
            Pr(X > x) = (x / xmin)^(-(alpha - 1)), x >= xmin

        Method:
            - choose candidate xmin only from sample values
            - optionally downsample candidate xmins to xmin_num points
            - for each xmin, fit alpha by MLE
            - compute KS distance between empirical tail CDF and model CDF
            - choose xmin, alpha minimizing KS
            - plot empirical CCDF, with x < xmin and x >= xmin in different colors
        """
        x = np.asarray(data, dtype=float).ravel()
        x = x[np.isfinite(x)]
        x = x[x > 0]

        if x.size == 0:
            raise ValueError("No positive finite data points found.")

        x.sort()
        x_min_data = float(np.min(x))
        x_max_data = float(np.max(x))

        if xmin_range is None:
            scan_low, scan_high = x_min_data, x_max_data
        else:
            scan_low, scan_high = map(float, xmin_range)

        if scan_low <= 0:
            raise ValueError("xmin_range lower bound must be > 0.")
        if scan_high <= scan_low:
            raise ValueError("xmin_range upper bound must be greater than lower bound.")

        # ---- candidate xmin: only choose from sample values ----
        unique_x = np.unique(x)
        candidate_xmins = unique_x[(unique_x >= scan_low) & (unique_x <= scan_high)]

        if candidate_xmins.size == 0:
            raise ValueError("No sample values fall inside xmin_range.")

        # logarithmically downsample candidates if too many
        if candidate_xmins.size > xmin_num:
            log_pos = np.linspace(
                0, np.log10(candidate_xmins.size - 1), int(xmin_num)
            )
            idx = np.unique(np.round(10**log_pos).astype(int))
            idx[0] = 0
            idx[-1] = candidate_xmins.size - 1
            candidate_xmins = candidate_xmins[idx]

        best = {
            "xmin": None,
            "alpha": None,
            "ks": np.inf,
            "n_tail": 0
        }

        # ---- fit alpha and KS for each xmin ----
        for xmin in candidate_xmins:
            tail = x[x >= xmin]
            n = tail.size

            if n < min_tail_size:
                continue

            logs = np.log(tail / xmin)
            denom = np.sum(logs)

            if denom <= 0:
                continue

            alpha = 1.0 + n / denom

            # Empirical tail CDF
            tail_sorted = np.sort(tail)
            empirical_cdf = np.arange(1, n + 1) / n

            # Model CDF for continuous power law
            model_cdf = 1.0 - (tail_sorted / xmin) ** (-(alpha - 1.0))

            ks = np.max(np.abs(empirical_cdf - model_cdf))

            if ks < best["ks"]:
                best["xmin"] = float(xmin)
                best["alpha"] = float(alpha)
                best["ks"] = float(ks)
                best["n_tail"] = int(n)

        if best["xmin"] is None:
            raise RuntimeError(
                "No valid xmin found. Try reducing min_tail_size or adjusting xmin_range/xmin_num."
            )

        xmin_hat = best["xmin"]
        alpha_hat = best["alpha"]
        ks_hat = best["ks"]

        # ---- empirical CCDF ----
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6), dpi=300)

        x_sorted = np.sort(x)
        n_all = x_sorted.size
        empirical_ccdf = (n_all - np.arange(1, n_all + 1)) / n_all

        # body: x < xmin
        body_mask = (x_sorted < xmin_hat) & (empirical_ccdf > 0)
        ax.plot(
            x_sorted[body_mask],
            empirical_ccdf[body_mask],
            marker="o",
            linestyle="None",
            markersize=3,
            color=color_body,
            alpha=0.7,
            label=r"Empirical CCDF ($x < x_{\min}$)"
        )

        # tail: x >= xmin
        tail_mask = (x_sorted >= xmin_hat) & (empirical_ccdf > 0)
        ax.plot(
            x_sorted[tail_mask],
            empirical_ccdf[tail_mask],
            marker="o",
            linestyle="None",
            markersize=3,
            color=color_tail,
            alpha=0.9,
            label=r"Empirical CCDF ($x \geq x_{\min}$)"
        )

        # ---- fitted CCDF on tail ----
        # ---- fitted CCDF on tail ----
        tail_data = np.sort(x[x >= xmin_hat])
        end_idx = int(np.ceil(0.999 * len(tail_data))) - 1
        x_fit_end = float(tail_data[end_idx])

        x_fit = np.logspace(np.log10(xmin_hat), np.log10(x_fit_end), 500)
        ccdf_fit = (x_fit / xmin_hat) ** (-(alpha_hat - 1.0))

        # To align visually with empirical whole-sample CCDF, scale by tail fraction
        tail_fraction = best["n_tail"] / n_all
        ccdf_fit_allscale = tail_fraction * ccdf_fit

        ax.plot(
            x_fit,
            ccdf_fit_allscale,
            linestyle="--",
            linewidth=2,
            color=color_fit,
            label=rf"Power-law fit ($\alpha={alpha_hat:.4f}$)"
        )

        # ---- mark xmin ----
        tail_idx = np.where((x_sorted >= xmin_hat) & (empirical_ccdf > 0))[0]

        if tail_idx.size > 0:
            idx0 = tail_idx[0]
            x_xmin_point = x_sorted[idx0]
            y_xmin_point = empirical_ccdf[idx0]
        else:
            # fallback: use fitted curve value at xmin
            x_xmin_point = xmin_hat
            y_xmin_point = tail_fraction

        ax.scatter(
            [x_xmin_point],
            [y_xmin_point],
            s=80,
            color=color_xmin,
            edgecolors="white",
            linewidths=0,
            zorder=5,
            label=rf"$x_{{\min}}={xmin_hat:.4g}$"
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=14)
        if show_legend:
            ax.legend(loc="upper right", frameon=False)

        print(f"Estimated xmin = {xmin_hat:.6g}")
        print(f"Estimated alpha = {alpha_hat:.6f}")
        print(f"KS distance = {ks_hat:.6f}")
        print(f"Tail sample size = {best['n_tail']}")

        if show:
            plt.show()

        if return_details:
            return {
                "xmin": xmin_hat,
                "alpha": alpha_hat,
                "ks": ks_hat,
                "n_tail": best["n_tail"],
                "candidate_xmins": candidate_xmins,
                "ax": ax
            }

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