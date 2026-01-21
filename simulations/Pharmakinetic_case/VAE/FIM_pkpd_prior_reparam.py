import jax
from jax import numpy as jnp
import diffrax
import equinox as eqx

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import jax.numpy as jnp
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def log_likelihood(params, epsilons, y, times, timepoints=6, T=10):

    class PKPD_ode(eqx.Module):
        # @jit
        def __call__(self, t, x, args): 
            phi1, phi2, time_scale = args
            param = jnp.multiply(jnp.array([[-phi1, phi2], [0, -phi2]]), time_scale)
            return param @ x
    
    term = ODETerm(PKPD_ode())
    solver = Tsit5()
    t0 = 0
    t1 = T
    dt0 = 0.01
    y0 = jnp.array([2.0, 3.0])
    saveat = SaveAt(ts=times)

    b_i = jnp.exp(params[2]) * epsilons
    phi1 = jnp.exp(params[0]) * jnp.exp(b_i)

    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, 
                      args=(phi1, jnp.exp(params[1]), 1.0), 
                      saveat=saveat,
                      adjoint=diffrax.ForwardMode())

    sigma0 = jnp.exp(params[-1])
    # sigma0 = 0.2 ** 2
    const = 2. * jnp.pi
    output = sol.ys[:,0]

    log_pxz = -0.5 * timepoints * jnp.log(const) - 0.5 * timepoints * jnp.log(sigma0) - 0.5 *(1/sigma0) * jnp.sum((output - y)**2)

    return log_pxz


grad_ll_fwd = jax.jacfwd(log_likelihood, argnums=0)                 # ∂/∂theta, forward-mode
hess_ll_fwd = jax.jacfwd(jax.jacfwd(log_likelihood, argnums=0),     # ∂²/∂theta², forward-forward
                         argnums=0)

def variance_estimation_2(all_datas, times, model_params, hessian=True, b_sample_size=50, ridge=1e-6):
    theta = jnp.asarray(model_params)
    N_subject = all_datas.shape[0]

    fim_list = []
    eps = jnp.asarray(np.random.normal(size=b_sample_size), dtype=theta.dtype)  # (S,)

    for i in range(N_subject):

        ll_s = jax.vmap(log_likelihood, in_axes=(None, 0, None, None))(theta, eps, all_datas[i,:], times[i,:])  # (S,)
        w = jax.nn.softmax(ll_s)  # (S,)

        g_s = jax.vmap(grad_ll_fwd, in_axes=(None, 0, None, None))(theta, eps, all_datas[i,:], times[i,:])      # (S,P)
        score = jnp.sum(w[:, None] * g_s, axis=0)  # (P,)

        if hessian:
            H_s = jax.vmap(hess_ll_fwd, in_axes=(None, 0, None, None))(theta, eps, all_datas[i,:], times[i,:])   # (S,P,P)

            EwH   = jnp.sum(w[:, None, None] * H_s, axis=0)  # (P,P)
            EwggT = jnp.sum(w[:, None, None] * (g_s[:,:,None] * g_s[:,None,:]), axis=0)  # (P,P)

            H_logm = EwH + EwggT - jnp.outer(score, score)
            I_i = -H_logm
        else:
            # OPG/BHHH contribution (Fisher approx when summed over subjects)
            I_i = jnp.outer(score, score)

        I_i = 0.5 * (I_i + I_i.T) + ridge * jnp.eye(I_i.shape[0], dtype=I_i.dtype)

        if not jnp.all(jnp.isfinite(I_i)):
            raise ValueError(f"Non-finite info matrix at subject {i}")

        fim_list.append(I_i)

    FIM = jnp.sum(jnp.stack(fim_list, axis=0), axis=0)
    FIM = 0.5 * (FIM + FIM.T) + ridge * jnp.eye(FIM.shape[0], dtype=FIM.dtype)

    FIM_np = np.asarray(FIM, dtype=np.float64)
    eig = np.linalg.eigvalsh(FIM_np)
    min_e = eig.min()
    cond = np.inf if min_e <= 0 else eig.max() / min_e
    print("min eigenvalue:", min_e, "cond:", cond)

    return np.linalg.pinv(FIM_np)

def variance_estimation(all_datas, times, model_params, hessian=True, b_sample_size=50):

    observed_FIM_list = []
    num_params = len(model_params)
    N_subject = 100
    epsilons = np.random.normal(loc=0, scale=1, size = b_sample_size)

    for i in range(N_subject):        
        log_likelihoods = jax.vmap(log_likelihood, in_axes=(None,0,None,None))(jnp.array(model_params), epsilons, all_datas[i,:], times[i, :])
        grad_log_likelihood_Y_b = jax.jacfwd(jax.vmap(log_likelihood, in_axes=(None,0,None,None)))(jnp.array(model_params), epsilons, all_datas[i,:], times[i, :])
        
        # likelihood_Y = jnp.exp(log_likelihoods) # p(Y|b)
        # grad_marginal_Y = jnp.mean(jnp.einsum('ij,i ->ij', grad_log_likelihood_Y_b, likelihood_Y), axis=0)
        # marginal_Y = jnp.mean(likelihood_Y)

        # Stable weights w via softmax(loglik)
        w = jax.nn.softmax(log_likelihoods)  # (S,)
        # score s = grad log p(y_i)
        score = jnp.sum(w[:, None] * grad_log_likelihood_Y_b, axis=0)  # (P,)

        if hessian:
            hessian_log_likelihood_Y_b = jax.jacfwd(jax.jacfwd
                                                    (jax.vmap(log_likelihood, in_axes=(None,0,None,None))))(jnp.array(model_params), epsilons,  all_datas[i,:], times[i,:])
            # E_w[H_s]
            EwH = jnp.sum(w[:, None, None] * hessian_log_likelihood_Y_b, axis=0)   # (P,P)

            # E_w[g_s g_s^T]
            ggT = grad_log_likelihood_Y_b[:, :, None] * grad_log_likelihood_Y_b[:, None, :]      # (S,P,P)
            EwggT = jnp.sum(w[:, None, None] * ggT, axis=0)    # (P,P)
            H_logm = EwH + EwggT - jnp.outer(score, score)
            I = -H_logm

            # Force symmetry (important before eig)
            I = 0.5 * (I + I.T)

            # Regularize a bit (MC noise can make it near-singular/indefinite)
            I = I + 1e-6 * jnp.eye(I.shape[0], dtype=I.dtype)

            observed_FIM_list.append(I)
        
        if not hessian:
            observed_information = score[:, None] * score[None, :]
            observed_FIM_list.append(observed_information)

    print(np.array(observed_FIM_list).shape)
    FIM = np.sum(np.array(observed_FIM_list), axis=0)
    # FIM = 0.5 * (FIM + FIM.T)
    eig = np.linalg.eigvalsh(FIM)
    print("min eigenvalue:", eig.min(), "cond:", eig.max()/eig.min())
    variance_matrix = np.linalg.pinv(FIM)
    
    return variance_matrix

# -----------------------------
# 1) Mean prediction function: Gauss-Newton
# -----------------------------
class PKPD_ode(eqx.Module):
    def __call__(self, t, x, args):
        phi1, phi2, time_scale = args
        A = jnp.multiply(jnp.array([[-phi1, phi2],
                                    [0.0, -phi2]]), time_scale)
        return A @ x

def mean_pred(params, eps, times, T=10.0):
    """
    params: (4,) = [log_phi1_pop, log_phi2, log_omega, log_sigma2]
            log_sigma2 does not affect the mean; that's OK (its Jacobian column is 0).
    eps: scalar ~ N(0,1)
    returns mu: (n_timepoints,) predicted x1(t) at 'times'
    """
    term = ODETerm(PKPD_ode())
    solver = Tsit5()
    t0, t1, dt0 = 0.0, T, 0.01
    y0 = jnp.array([2.0, 3.0])
    saveat = SaveAt(ts=times)

    omega = jnp.exp(params[2])             # SD of random effect
    b = omega * eps
    phi1 = jnp.exp(params[0]) * jnp.exp(b) # lognormal random effect on phi1
    phi2 = jnp.exp(params[1])

    sol = diffeqsolve(
        term, solver, t0, t1, dt0, y0,
        args=(phi1, phi2, 1.0),
        saveat=saveat,
        adjoint=diffrax.ForwardMode()
    )
    mu = sol.ys[:, 0]  # observe x1
    return mu


# Jacobian of mean wrt params (ForwardMode-safe)
jac_mu = jax.jacfwd(mean_pred, argnums=0)


# -----------------------------
# 2) Main variance estimation
# -----------------------------
def variance_estimation_gn(
    all_datas, times, params, b_sample_size=50, T=10.0,
    ridge=1e-6, use_sandwich=True, seed=0
):
    """
    all_datas: (N, n) observed y
    times:     (N, n) observation times (can vary by subject)
    params:    (4,) [log_phi1_pop, log_phi2, log_omega, log_sigma2]
              log_sigma2 = log(variance) in your likelihood.
    Returns:
      cov_gn, cov_sandwich (if use_sandwich), plus eigen diagnostics.
    """
    params = jnp.asarray(params)
    N, n = all_datas.shape
    P = params.shape[0]

    # Important: fix epsilons (common random numbers) for stability
    rng = np.random.default_rng(seed)
    eps_all = jnp.asarray(rng.normal(size=(N, b_sample_size)), dtype=params.dtype)

    H = jnp.zeros((P, P), dtype=params.dtype)  # GN/Fisher-scoring "Hessian" approximation
    S = jnp.zeros((P, P), dtype=params.dtype)  # OPG of marginal scores (for sandwich)

    # vmaps over S samples for one subject
    v_mu  = jax.vmap(mean_pred, in_axes=(None, 0, None, None))   # -> (S, n)
    v_J   = jax.vmap(jac_mu,   in_axes=(None, 0, None, None))    # -> (S, n, P)

    for i in range(N):
        y_i = jnp.asarray(all_datas[i, :], dtype=params.dtype)
        t_i = jnp.asarray(times[i, :],     dtype=params.dtype)
        eps = eps_all[i, :]  # (S,)

        mu_s = v_mu(params, eps, t_i, T)      # (S, n)
        J_s  = v_J(params, eps, t_i, T)       # (S, n, P)

        r_s = y_i[None, :] - mu_s             # (S, n)
        rss_s = jnp.sum(r_s**2, axis=1)        # (S,)

        sigma2 = jnp.exp(params[-1])          # variance
        # conditional Gaussian loglik per sample (up to const)
        ll_s = -0.5 * n * jnp.log(2.0 * jnp.pi) - 0.5 * n * jnp.log(sigma2) - 0.5 * rss_s / sigma2

        # stable weights for the MC marginalization
        w = jax.nn.softmax(ll_s)              # (S,)

        # ---- Score (marginal) using only first derivatives ----
        # conditional score for mean params: g_s = (1/sigma2) J^T r
        # Shape: (S,P)
        g_s = (1.0 / sigma2) * jnp.einsum("snp,sn->sp", J_s, r_s)

        # conditional score for log(sigma2): u_s = d/d log(sigma2) ll_s
        # ll = -0.5 n log(sigma2) - 0.5 rss/sigma2 + const
        # d/d log(sigma2) = sigma2 d/d sigma2 -> u = -0.5 n + 0.5 rss/sigma2
        u_s = -0.5 * n + 0.5 * rss_s / sigma2  # (S,)

        # marginal score = weighted average
        score = jnp.sum(w[:, None] * g_s, axis=0)     # (P,)
        score = score.at[-1].set(jnp.sum(w * u_s))    # set sigma2 component properly

        # accumulate OPG for sandwich
        if use_sandwich:
            S = S + jnp.outer(score, score)

        # ---- Gauss–Newton / Fisher-scoring information ----
        # GN for mean parameters: (1/sigma2) J^T J (per sample), then weighted average
        JtJ_s = (1.0 / sigma2) * jnp.einsum("snp,snq->spq", J_s, J_s)  # (S,P,P)
        H_i = jnp.sum(w[:, None, None] * JtJ_s, axis=0)               # (P,P)

        # For log(sigma2) we add its Fisher information (conditional normal):
        # per subject (n obs): I(log sigma2) = n/2
        # (and cross-terms with mean params are ~0 under Gaussian models)
        H_i = H_i.at[-1, -1].add(0.5 * n)

        # symmetrize and ridge
        H_i = 0.5 * (H_i + H_i.T) + ridge * jnp.eye(P, dtype=params.dtype)

        H = H + H_i

    # Final sym + ridge
    H = 0.5 * (H + H.T) + ridge * jnp.eye(P, dtype=params.dtype)

    H_np = np.asarray(H, dtype=np.float64)
    eig = np.linalg.eigvalsh(H_np)
    min_e = eig.min()
    cond = np.inf if min_e <= 0 else eig.max() / min_e

    cov_gn = np.linalg.pinv(H_np)

    if not use_sandwich:
        return cov_gn, {"min_eig": float(min_e), "cond": float(cond)}

    S_np = np.asarray(0.5 * (S + S.T), dtype=np.float64)
    cov_sandwich = cov_gn @ S_np @ cov_gn

    return cov_gn, cov_sandwich, {"min_eig": float(min_e), "cond": float(cond)}


if __name__ == "__main__":

    params = pd.read_csv("results/PKPD_irregular_5p_100_10d_noise.txt", sep=" ",header=None).values
    b_sample_size = 5000
    hessian=False
    folder = "EXPs/pkpd_variance"
    
    for i in range(99):
        try:

            data_file = 'PKPD_datasets/5_irregular_1latent_theta1/dataset_'+str(i+1)+'.npy'
            all_datas = np.load(data_file, allow_pickle=True)
            df = pd.DataFrame(all_datas.tolist())
            y = np.stack(df['Y'].to_numpy())
            times = np.stack(df['t'].to_numpy())
           
            variance_matrix = variance_estimation(y, times, params[i, :], hessian=False, b_sample_size=3000)
           
            from pathlib import Path
            if hessian:
                result_folder = folder +'_'+str(b_sample_size)
            else:
                result_folder = folder +'_nohessian_'+str(b_sample_size)

            Path(result_folder).mkdir(parents=True, exist_ok=True)
            np.save(result_folder+'/variance_matrix_'+str(i+1), variance_matrix)

        except Exception as error :
            print(error)

    

    
    

    

