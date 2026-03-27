import os
import ot
import time
import torch
import warnings
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union, List
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from .utils import fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix

# =====================================================================
# STFA PIPELINE: STAGE 1 (Intrinsic Representations)
# =====================================================================

def compute_adaptive_knn_graph(
    spatial_coords: np.ndarray, 
    min_k: int = 10, 
    max_k: int = 40, 
    target_mass: float = 0.95
) -> Tuple[sp.csr_matrix, np.ndarray, int]:
    '''Adaptive minimal k-NN graph yielding dominant connected component.'''
    n_cells = spatial_coords.shape[0]
    optimal_k = max_k
    dominant_mask = np.ones(n_cells, dtype=bool)
    best_A = None
    
    for k in range(min_k, max_k + 1, 5):
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        row = np.repeat(np.arange(n_cells), k)
        col = indices[:, 1:].flatten()
        data = np.ones(len(row))
        
        A_dir = sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
        A_mut = A_dir.multiply(A_dir.T)
        
        n_components, labels = connected_components(csgraph=A_mut, directed=False, return_labels=True)
        counts = np.bincount(labels)
        dominant_label = np.argmax(counts)
        mass = counts[dominant_label] / n_cells
        
        if mass >= target_mass:
            optimal_k = k
            dominant_mask = (labels == dominant_label)
            best_A = A_mut
            break
            
    if best_A is None:
        optimal_k = max_k
        nbrs = NearestNeighbors(n_neighbors=optimal_k+1, metric='euclidean').fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        row = np.repeat(np.arange(n_cells), optimal_k)
        col = indices[:, 1:].flatten()
        data = np.ones(len(row))
        A_dir = sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
        best_A = A_dir.multiply(A_dir.T)
        n_components, labels = connected_components(csgraph=best_A, directed=False, return_labels=True)
        counts = np.bincount(labels)
        dominant_label = np.argmax(counts)
        dominant_mask = (labels == dominant_label)
        
    return best_A, dominant_mask, optimal_k

def compute_spectral_diffusion(
    A: sp.csr_matrix, 
    dominant_mask: np.ndarray, 
    cell_types: np.ndarray
) -> Tuple[np.ndarray, float, List[int]]:
    '''Discrete spectral diffusion signatures spanning {1, tau/4, tau} scales.'''
    n_cells = A.shape[0]
    A_sub = A[dominant_mask][:, dominant_mask]
    n_sub = A_sub.shape[0]
    
    degrees = np.array(A_sub.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    L_sym = sp.eye(n_sub) - D_inv_sqrt @ A_sub @ D_inv_sqrt
    
    try:
        from scipy.sparse.linalg import eigsh
        evals, evecs = eigsh(L_sym, k=2, which='SM')
        lambda_2 = evals[1] if evals[1] > 1e-10 else 1e-5
    except Exception:
        lambda_2 = 1e-2
        
    tau_mix = 1.0 / lambda_2
    scales = [1, max(1, int(tau_mix / 4)), max(1, int(tau_mix))]
    
    D_inv = sp.diags(1.0 / np.maximum(degrees, 1e-12))
    P = D_inv @ A_sub
    
    sub_types = cell_types[dominant_mask]
    encoder = OneHotEncoder(sparse_output=False)
    X = encoder.fit_transform(sub_types.reshape(-1, 1))
    
    signatures = []
    current_H = X
    current_t = 0
    
    for t in scales:
        steps = t - current_t
        for _ in range(steps):
            current_H = P @ current_H
        current_t = t
        signatures.append(current_H)
        
    M_topo_features = np.hstack(signatures)
    return M_topo_features, tau_mix, scales

def compute_boundary_uncertainty(
    spatial_coords: np.ndarray, 
    dominant_mask: np.ndarray, 
    A: sp.csr_matrix
) -> np.ndarray:
    '''Smooth boundary uncertainty.'''
    n_cells = spatial_coords.shape[0]
    u = np.zeros(n_cells)
    
    A_sub = A[dominant_mask][:, dominant_mask]
    degrees = np.array(A_sub.sum(axis=1)).flatten()
    
    threshold_deg = np.percentile(degrees, 10)
    periphery_idx = np.where(degrees <= threshold_deg)[0]
    
    if len(periphery_idx) == 0:
        u[dominant_mask] = 1.0
        return u
        
    n_sub = A_sub.shape[0]
    # Build an explicit (n_sub + 1) x (n_sub + 1) augmented graph with a dummy
    # node connected to every low-degree periphery node.
    A_sub_augmented = sp.lil_matrix((n_sub + 1, n_sub + 1), dtype=A_sub.dtype)
    A_sub_augmented[:n_sub, :n_sub] = A_sub
    A_sub_augmented[periphery_idx, n_sub] = 1
    A_sub_augmented[n_sub, periphery_idx] = 1
    A_sub_augmented = A_sub_augmented.tocsr()
    
    dist_matrix = shortest_path(csgraph=A_sub_augmented, directed=False, indices=n_sub, unweighted=True)
    d_i = dist_matrix[:n_sub]
    
    sigma_d = np.percentile(d_i, 5) if np.max(d_i) > 0 else 1.0
    sigma_d = max(sigma_d, 1.0)
    
    u_sub = 1 - 0.5 * np.exp(-d_i / sigma_d)
    u[dominant_mask] = u_sub
    return u

# =====================================================================
# STFA PIPELINE: STAGE 2 (Minimal Coarse Anchor Matching)
# =====================================================================

def compute_community_anchors(
    A_A: sp.csr_matrix, mask_A: np.ndarray, M_topo_A: np.ndarray, u_A: np.ndarray, X_A: np.ndarray,
    A_B: sp.csr_matrix, mask_B: np.ndarray, M_topo_B: np.ndarray, u_B: np.ndarray, X_B: np.ndarray,
    n_clusters: int = 15
) -> np.ndarray:
    nA, nB = len(mask_A), len(mask_B)
    pi_comm_full = np.zeros((nA, nB))
    
    A_sub_A = A_A[mask_A][:, mask_A]
    A_sub_B = A_B[mask_B][:, mask_B]
    n_sub_A, n_sub_B = A_sub_A.shape[0], A_sub_B.shape[0]
    
    sc_A = SpectralClustering(n_clusters=min(n_clusters, n_sub_A), affinity='precomputed', random_state=42)
    labels_sub_A = sc_A.fit_predict(A_sub_A)
    
    sc_B = SpectralClustering(n_clusters=min(n_clusters, n_sub_B), affinity='precomputed', random_state=42)
    labels_sub_B = sc_B.fit_predict(A_sub_B)
    
    n_c_A, n_c_B = len(np.unique(labels_sub_A)), len(np.unique(labels_sub_B))
    
    n_types_A = X_A.shape[1]; tau_mix_idx_start_A = n_types_A * 2
    n_types_B = X_B.shape[1]; tau_mix_idx_start_B = n_types_B * 2
    
    desc_A, mass_A = [], []
    for c in range(n_c_A):
        c_mask = (labels_sub_A == c)
        mass = c_mask.sum() / n_sub_A
        mass_A.append(mass)
        h_tau = M_topo_A[c_mask, tau_mix_idx_start_A:].mean(axis=0) if tau_mix_idx_start_A < M_topo_A.shape[1] else M_topo_A[c_mask].mean(axis=0)
        ctype = X_A[mask_A][c_mask].mean(axis=0)
        bndry = u_A[mask_A][c_mask].mean()
        desc_A.append(np.concatenate([h_tau, ctype, [mass, bndry]]))
        
    desc_B, mass_B = [], []
    for c in range(n_c_B):
        c_mask = (labels_sub_B == c)
        mass = c_mask.sum() / n_sub_B
        mass_B.append(mass)
        h_tau = M_topo_B[c_mask, tau_mix_idx_start_B:].mean(axis=0) if tau_mix_idx_start_B < M_topo_B.shape[1] else M_topo_B[c_mask].mean(axis=0)
        ctype = X_B[mask_B][c_mask].mean(axis=0)
        bndry = u_B[mask_B][c_mask].mean()
        desc_B.append(np.concatenate([h_tau, ctype, [mass, bndry]]))
        
    desc_A = np.array(desc_A); desc_B = np.array(desc_B)
    mass_A = np.array(mass_A); mass_B = np.array(mass_B)
    min_dim = min(desc_A.shape[1], desc_B.shape[1])
    desc_A = desc_A[:, :min_dim]; desc_B = desc_B[:, :min_dim]
    
    C_comm = euclidean_distances(desc_A, desc_B)
    if C_comm.max() > 0: C_comm /= C_comm.max()
    
    try:
        pi_c = ot.unbalanced.sinkhorn_unbalanced(mass_A, mass_B, C_comm, reg=0.05, reg_m=0.1)
    except Exception:
        pi_c = np.outer(mass_A, mass_B)
        
    pi_sub = np.zeros((n_sub_A, n_sub_B))
    for i in range(n_c_A):
        c_mask_A = (labels_sub_A == i)
        for j in range(n_c_B):
            c_mask_B = (labels_sub_B == j)
            val = pi_c[i, j]
            count_A, count_B = c_mask_A.sum(), c_mask_B.sum()
            if count_A > 0 and count_B > 0:
                row_idx = np.where(c_mask_A)[0]
                col_idx = np.where(c_mask_B)[0]
                pi_sub[np.ix_(row_idx, col_idx)] = val / (count_A * count_B)
                
    idx_A = np.where(mask_A)[0]
    idx_B = np.where(mask_B)[0]
    pi_comm_full[np.ix_(idx_A, idx_B)] = pi_sub
    return pi_comm_full

# =====================================================================
# STFA PIPELINE: STAGE 3 (Bilevel Gamma Calibration)
# =====================================================================

def _normalize_cost_matrix(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(M)
    if min_val < 0:
        M = M - min_val
    max_val = np.max(M)
    if max_val > 0:
        M = M / max_val
    return M

def _sanitize_distribution(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).copy()
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights[weights < 0] = 0.0
    total = weights.sum()
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total

def _ensure_nonempty_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if not np.any(mask):
        mask[np.random.randint(len(mask))] = True
    return mask

def solve_unbalanced_fgw(
    C1: np.ndarray, C2: np.ndarray, M: np.ndarray, 
    p: np.ndarray, q: np.ndarray, 
    gamma: float, alpha: float, 
    reg: float = 0.01
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    active_p = p > 0
    active_q = q > 0

    if not np.any(active_p) or not np.any(active_q):
        return np.outer(_sanitize_distribution(p), _sanitize_distribution(q))

    C1_sub = _normalize_cost_matrix(C1[np.ix_(active_p, active_p)])
    C2_sub = _normalize_cost_matrix(C2[np.ix_(active_q, active_q)])
    M_sub = _normalize_cost_matrix(M[np.ix_(active_p, active_q)])
    p_sub = _sanitize_distribution(p[active_p])
    q_sub = _sanitize_distribution(q[active_q])

    reg_candidates = sorted({max(float(reg), 0.05), 0.1, 0.2})
    pi_sub = None

    for epsilon in reg_candidates:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*numerical errors.*")
                warnings.filterwarnings("error", message=".*failed to produce a transport plan.*")
                warnings.filterwarnings("error", category=RuntimeWarning)
                candidate = ot.gromov.entropic_fused_gromov_wasserstein(
                    M_sub, C1_sub, C2_sub, p_sub, q_sub,
                    loss_fun='square_loss', epsilon=epsilon, alpha=alpha, numItermax=100
                )
            if np.all(np.isfinite(candidate)) and candidate.sum() > 0:
                pi_sub = np.maximum(candidate, 0.0)
                pi_sub /= pi_sub.sum()
                break
        except Exception:
            continue

    if pi_sub is None:
        pi_sub = np.outer(p_sub, q_sub)

    pi = np.zeros_like(M, dtype=float)
    pi[np.ix_(active_p, active_q)] = pi_sub
    return pi

def compute_bilevel_gamma_calibration(
    C1: np.ndarray, C2: np.ndarray, M_topo: np.ndarray, M_gene: np.ndarray, M_anchor: np.ndarray,
    p: np.ndarray, q: np.ndarray, alpha: float
) -> float:
    nA, nB = len(p), len(q)
    N_sub = min(500, nA, nB)
    
    idx_A = np.random.choice(nA, N_sub, replace=False)
    idx_B = np.random.choice(nB, N_sub, replace=False)
    
    C1_sub = C1[np.ix_(idx_A, idx_A)]
    C2_sub = C2[np.ix_(idx_B, idx_B)]
    M_t_sub = M_topo[np.ix_(idx_A, idx_B)]
    M_g_sub = M_gene[np.ix_(idx_A, idx_B)]
    M_a_sub = M_anchor[np.ix_(idx_A, idx_B)]
    p_sub = np.ones(N_sub) / N_sub
    q_sub = np.ones(N_sub) / N_sub
    
    dropout_A = _ensure_nonempty_mask(np.random.rand(N_sub) > 0.05)
    dropout_B = _ensure_nonempty_mask(np.random.rand(N_sub) > 0.05)
    
    def evaluate_gamma(g: float) -> float:
        M_sub = g * M_g_sub + (1 - g) * M_t_sub + 0.1 * M_a_sub
        pi_base = solve_unbalanced_fgw(C1_sub, C2_sub, M_sub, p_sub, q_sub, g, alpha)
        
        p_drop = p_sub.copy()
        q_drop = q_sub.copy()
        p_drop[~dropout_A] = 0
        q_drop[~dropout_B] = 0
        p_drop = _sanitize_distribution(p_drop)
        q_drop = _sanitize_distribution(q_drop)
        
        pi_drop = solve_unbalanced_fgw(C1_sub, C2_sub, M_sub, p_drop, q_drop, g, alpha)
        diff = pi_base[np.ix_(dropout_A, dropout_B)] - pi_drop[np.ix_(dropout_A, dropout_B)]
        return float(np.linalg.norm(diff, 'fro'))

    gammas = np.linspace(0.1, 0.9, 9)
    scores = [evaluate_gamma(g) for g in gammas]
    return float(gammas[np.argmin(scores)])

# =====================================================================
# STFA PIPELINE: STAGE 4 (Robust Deformation Refinement)
# =====================================================================

def refine_deformation(
    coords_A: np.ndarray, coords_B: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    pi_flat = pi.flatten()
    pi_nonzero = pi_flat[pi_flat > 1e-8]
    if len(pi_nonzero) > 0:
        try:
            from skimage.filters import threshold_otsu
            theta = threshold_otsu(pi_nonzero)
        except Exception:
            theta = pi_nonzero.mean()
    else:
        theta = 0.0
        
    inliers = np.argwhere(pi > theta)
    if len(inliers) < 10: return coords_A
        
    if len(inliers) > 500:
        inliers = inliers[np.random.choice(len(inliers), 500, replace=False)]
        
    src_pts = coords_A[inliers[:, 0]]
    dst_pts = coords_B[inliers[:, 1]]
    
    try:
        from scipy.interpolate import RBFInterpolator
        displacement = dst_pts - src_pts
        rbf = RBFInterpolator(src_pts, displacement, kernel='thin_plate_spline')
        
        disp_full = rbf(coords_A)
        max_disp = np.percentile(np.linalg.norm(displacement, axis=1), 95)
        disp_norms = np.linalg.norm(disp_full, axis=1)
        clip_mask = disp_norms > max_disp * 1.5
        
        if np.any(clip_mask):
            factors = (max_disp * 1.5) / np.maximum(disp_norms[clip_mask], 1e-12)
            disp_full[clip_mask] = disp_full[clip_mask] * factors[:, None]
        
        warped_A = coords_A + 0.5 * disp_full
        return warped_A
    except Exception as e:
        print(f"TPS Failed: {e}")
        return coords_A

# =====================================================================
# MAIN PIPELINE WRAPPER
# =====================================================================

def pairwise_align(
    sliceA: AnnData, sliceB: AnnData, 
    alpha: float, beta: float, gamma: float, radius: float,
    filePath: str, use_rep: Optional[str] = None, 
    G_init = None, a_distribution = None, b_distribution = None, 
    norm: bool = False, numItermax: int = 6000, 
    backend = ot.backend.NumpyBackend(), use_gpu: bool = False, 
    return_obj: bool = False, verbose: bool = False, 
    gpu_verbose: bool = True, sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None, overwrite = False,
    neighborhood_dissimilarity: str='jsd', dummy_cell: bool = True,
    **kwargs
) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    '''
    STFA Pipeline (Spatial Transcriptomics Alignment).
    Executes the 5-stage mathematically robust STFA framework.
    '''
    start_time = time.time()
    if not os.path.exists(filePath): os.makedirs(filePath)
    
    with open(f"{filePath}/log.txt", "w") as logFile:
        logFile.write(f"pairwise_align_STFA\n{datetime.datetime.now()}\n")
        logFile.write(f"Executing robust STFA pipeline on {sliceA_name} and {sliceB_name}\n")

    # Filter shared cell types
    shared_cell_types = pd.Index(sliceA.obs['cell_type_annot']).unique().intersection(pd.Index(sliceB.obs['cell_type_annot']).unique())
    if len(shared_cell_types) == 0: raise ValueError("No shared cell types.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_cell_types)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_cell_types)]

    coords_A = sliceA.obsm['spatial'].copy()
    coords_B = sliceB.obsm['spatial'].copy()
    
    types_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    types_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    
    encoder = OneHotEncoder(sparse_output=False).fit(np.concatenate([types_A, types_B]).reshape(-1, 1))
    X_A = encoder.transform(types_A.reshape(-1, 1))
    X_B = encoder.transform(types_B.reshape(-1, 1))
    
    # --- STAGE 1: Intrinsic Representations ---
    A_A, mask_A, k_A = compute_adaptive_knn_graph(coords_A)
    A_B, mask_B, k_B = compute_adaptive_knn_graph(coords_B)
    
    M_topo_A_feat, tau_A, _ = compute_spectral_diffusion(A_A, mask_A, types_A)
    M_topo_B_feat, tau_B, _ = compute_spectral_diffusion(A_B, mask_B, types_B)
    
    u_A = compute_boundary_uncertainty(coords_A, mask_A, A_A)
    u_B = compute_boundary_uncertainty(coords_B, mask_B, A_B)
    
    dist_topo = cosine_distances(M_topo_A_feat, M_topo_B_feat)
    M_topo = np.ones((len(coords_A), len(coords_B)))
    idx_A = np.where(mask_A)[0]; idx_B = np.where(mask_B)[0]
    M_topo[np.ix_(idx_A, idx_B)] = dist_topo
    
    M_boundary = 1 - np.outer(u_A, u_B)
    
    # --- STAGE 2: Coarse Anchors ---
    pi_comm = compute_community_anchors(A_A, mask_A, M_topo_A_feat, u_A, X_A,
                                        A_B, mask_B, M_topo_B_feat, u_B, X_B)
    M_anchor = -np.log(pi_comm + 1e-5)
    
    # --- STAGE 3: M_gene and Calibration ---
    A_X = to_dense_array(extract_data_matrix(sliceA, use_rep))
    B_X = to_dense_array(extract_data_matrix(sliceB, use_rep))
    M_gene = cosine_distances(A_X + 0.01, B_X + 0.01)
    
    C_A = euclidean_distances(coords_A, coords_A)
    C_B = euclidean_distances(coords_B, coords_B)
    if norm:
        C_A /= (np.min(C_A[C_A > 0]) if len(C_A[C_A > 0]) else 1.0)
        C_B /= (np.min(C_B[C_B > 0]) if len(C_B[C_B > 0]) else 1.0)
        
    p = np.ones(len(coords_A)) / len(coords_A)
    q = np.ones(len(coords_B)) / len(coords_B)
    
    best_gamma = compute_bilevel_gamma_calibration(C_A, C_B, M_topo, M_gene, M_anchor, p, q, alpha)
    gamma_final = best_gamma if best_gamma is not None else gamma
    
    M_fused = gamma_final * M_gene + (1 - gamma_final) * M_topo + 0.1 * M_anchor + 0.05 * M_boundary
    
    # Optional GPU conversion for main FGW using POT/utils
    if isinstance(backend, ot.backend.TorchBackend) and use_gpu and torch.cuda.is_available():
        M_fused_t = torch.tensor(M_fused).cuda()
        C_A_t = torch.tensor(C_A).cuda()
        C_B_t = torch.tensor(C_B).cuda()
        p_t = torch.tensor(p).cuda()
        q_t = torch.tensor(q).cuda()
        pi_tensor, logw = fused_gromov_wasserstein_incent(
            M_fused_t, torch.zeros_like(M_fused_t).cuda(), C_A_t, C_B_t, p_t, q_t, 
            loss_fun='square_loss', alpha=alpha, gamma=gamma_final, log=True, numItermax=numItermax, use_gpu=True
        )
        pi = pi_tensor.cpu().numpy()
    else:
        M_fused_np = backend.from_numpy(M_fused)
        C_A_np = backend.from_numpy(C_A)
        C_B_np = backend.from_numpy(C_B)
        p_np = backend.from_numpy(p)
        q_np = backend.from_numpy(q)
        M2_dummy = np.zeros_like(M_fused)
        pi_tensor, logw = fused_gromov_wasserstein_incent(
            M_fused_np, backend.from_numpy(M2_dummy), C_A_np, C_B_np, p_np, q_np, 
            loss_fun='square_loss', alpha=alpha, gamma=gamma_final, log=True, numItermax=numItermax, use_gpu=False
        )
        pi = backend.to_numpy(pi_tensor)
        
    # --- STAGE 4: Deformation Refinement ---
    warped_coords_A = refine_deformation(coords_A, coords_B, pi)
    C_A_warped = euclidean_distances(warped_coords_A, warped_coords_A)
    
    if isinstance(backend, ot.backend.TorchBackend) and use_gpu and torch.cuda.is_available():
        pi_tensor, _ = fused_gromov_wasserstein_incent(
            M_fused_t, torch.zeros_like(M_fused_t).cuda(), torch.tensor(C_A_warped).cuda(), C_B_t, p_t, q_t, 
            loss_fun='square_loss', alpha=alpha, gamma=gamma_final, log=True, numItermax=numItermax, use_gpu=True
        )
        pi = pi_tensor.cpu().numpy()
    out_obj_gene = np.sum(M_gene * pi)
    out_obj_topo = np.sum(M_topo * pi)

    print(f"STFA Alignment complete in {time.time()-start_time:.2f}s using adaptive gamma={gamma_final:.2f}")
    if return_obj:
        return pi, out_obj_topo, out_obj_gene, out_obj_topo, out_obj_gene
    return pi
