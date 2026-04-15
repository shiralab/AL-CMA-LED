import cma
import numpy as np
import math
from scipy.stats import norm
from cma.evolution_strategy import CMAEvolutionStrategy
from cma.constraints_handler import AugmentedLagrangian

def calculate_al_penalty_terms(lam, mu, g):
    lam = np.asarray(lam).flatten()
    mu = np.asarray(mu).flatten()
    g = np.asarray(g).flatten()
    
    m = len(lam)
    P_k = np.zeros(m)
    
    for k in range(m):
        gk = g[k]
        lamk = lam[k]
        muk = mu[k]
        
        if muk < 1e-18:
            P_k[k] = lamk * gk 
            continue

        if gk >= -lamk / muk:
            P_k[k] = lamk * gk + 0.5 * muk * gk**2
        else:
            P_k[k] = - (lamk**2) / (2.0 * muk)
    
    return P_k

class LoggerCallback:
    def __init__(self, problem, log_filepath):
        self.problem = problem; self.log_file = open(log_filepath, 'w')
        self.header_written = False
        self.dim = problem.dimension

    def log(self, itr, total_evals, sigma, penalized_f, f, g, mean_penalized_f, mean_f, mean_g, v1, mean, N_eff, infeasible_prob, max_eigenvalue, mahalanobis_dists, lam, mu, f_diff, penalty_k_vector):
        if not self.header_written:
            self.n_cons = len(lam)
            v1_headers = ",".join([f"v1_{i}" for i in range(self.dim)])
            mean_headers = ",".join([f"mean_{i}" for i in range(self.dim)])
            maha_headers = ",".join([f"mahalanobis_dist_{i}" for i in range(1, self.n_cons + 1)])
            lam_headers = ",".join([f"lam_{i}" for i in range(self.n_cons)])
            mu_headers = ",".join([f"mu_{i}" for i in range(self.n_cons)])
            P_k_headers = ",".join([f"P_k_{i}" for i in range(self.n_cons)])
            
            header = f"itr,evals,sigma,best_penalized_f,best_f,best_g,mean_penalized_f,mean_f,mean_g,f_diff,{P_k_headers},{v1_headers},{mean_headers},{maha_headers},{lam_headers},{mu_headers},N_eff,infeasible_prob,max_eigenvalue\n"
            self.log_file.write(header)
            self.header_written = True

        v1_str = ",".join(map(str, v1.flatten()))
        mean_str = ",".join(map(str, mean.flatten()))
        maha_str = ",".join(map(str, mahalanobis_dists.flatten()))
        lam_str = ",".join(map(str, lam.flatten()))
        mu_str = ",".join(map(str, mu.flatten()))
        P_k_str = ",".join(map(str, penalty_k_vector.flatten()))
        
        log_line = f"{itr},{total_evals},{sigma},{penalized_f},{f},{g},{mean_penalized_f},{mean_f},{mean_g},{f_diff},{P_k_str},{v1_str},{mean_str},{maha_str},{lam_str},{mu_str},{N_eff},{infeasible_prob},{max_eigenvalue}\n"
        self.log_file.write(log_line)

    def close(self): self.log_file.close()

def initialize_lagrangian_parameters(F_values, G_values, dimension):
    delta_f = np.percentile(F_values, 75) - np.percentile(F_values, 25)
    if not G_values or G_values[0] is None: return np.array([]), np.array([])
    if len(G_values[0]) == 0: return np.array([]), np.array([])
    n_constraints = len(G_values[0]); lam = np.ones(n_constraints); mu = np.ones(n_constraints)
    for k in range(n_constraints):
        g_k_values = [g[k] for g in G_values if g is not None and len(g) > k]
        if not g_k_values: continue
        delta_g_k = np.percentile(g_k_values, 75) - np.percentile(g_k_values, 25)
        denominator_lam = dimension * delta_g_k
        if denominator_lam > 1e-9: lam[k] = abs(delta_f) / (denominator_lam + 1e-9)
        g_k_sq_values = [g**2 for g in g_k_values]; delta_g_k_sq = np.percentile(g_k_sq_values, 75) - np.percentile(g_k_sq_values, 25)
        denominator_mu = 5 * dimension * (delta_g_k + 1e-6 * delta_g_k_sq)
        if denominator_mu > 1e-9: mu[k] = 2 * abs(delta_f) / (denominator_mu + 1e-9)
    return lam, mu

def run_al_cmaes(problem, budget, initial_mean_generator=None, log_filepath=None):
    dim = problem.dimension; remaining_budget = budget; popsize_multiplier = 1.0; total_evals = 0
    al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5; al.chi_domega = 2**(1/(5*dim))
    is_al_initialized = False
    if log_filepath: logger = LoggerCallback(problem, log_filepath)
    else: logger = None
    try:
        while remaining_budget > 0:
            target_hit = False
            mean = (initial_mean_generator() if initial_mean_generator else np.random.uniform(-5, 5, (dim, 1))).reshape(dim, 1)
            sigma = 1.0; C = np.eye(dim); ps = np.zeros((dim, 1)); pc = np.zeros((dim, 1))
            lamb = int((4 + math.floor(3 * np.log(dim))) * popsize_multiplier)
            if remaining_budget < lamb: break
            mu = lamb // 2; wrh = np.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
            w = wrh / sum(wrh); mueff = 1 / np.sum(w**2); cm = 1.0
            evals_this_run = 0; itr = 0
            cs = (mueff + 2) / (dim + mueff + 5); ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            alpha_cov = min(2.0, lamb / 3.0); c1 = alpha_cov / ((dim + 1.3)**2 + mueff); cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + alpha_cov * mueff / 2))
            chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            while True:
                if evals_this_run + lamb > remaining_budget: break
                itr += 1
                eigens, evecs = np.linalg.eigh(C); eigens[eigens < 0] = 0; sqrtC = evecs @ np.diag(np.sqrt(eigens)) @ evecs.T
                z_vectors = np.random.randn(lamb, dim)
                solutions = (mean.T + sigma * (sqrtC @ z_vectors.T).T)
                F_values = [problem(s) for s in solutions]; G_values = [problem.constraint(s) for s in solutions]
                if not is_al_initialized:
                    al.lam, al.mu = initialize_lagrangian_parameters(F_values, G_values, dim)
                    is_al_initialized = True
                penalized_fitnesses = [f + sum(al(g)) for f, g in zip(F_values, G_values)]
                sorted_indices = np.argsort(penalized_fitnesses); best_idx = sorted_indices[0]
                current_total_evals = total_evals + evals_this_run + lamb
                evals_this_run += lamb

                if logger:
                    f_at_mean = problem(mean.flatten()); g_at_mean = problem.constraint(mean.flatten())
                    penalized_f_at_mean = f_at_mean + sum(al(g_at_mean))
                    g_vals = G_values[best_idx]; g_to_log = np.max(g_vals) if g_vals.size > 0 else 0.0
                    N_eff = dim; cov = sigma**2 * C
                    violating_samples = sum(1 for g_array in G_values if np.any(g_array > 0))
                    infeasible_prob = violating_samples / lamb
                    if hasattr(problem, 'f_opt'):
                        f_diff = f_at_mean - problem.f_opt
                    else:
                        f_diff = f_at_mean
                
                    penalty_k_vector = calculate_al_penalty_terms(al.lam, al.mu, g_at_mean)
                    
                    n_cons = len(al.lam) 
                    mahalanobis_dists = np.zeros(n_cons)
                    current_mean_flat = mean.flatten()
                    cov_diag = cov.diagonal()
                    
                    for i in range(n_cons): 
                        dim_index = i + 1 
                        if dim_index < dim:
                            mu_i = current_mean_flat[dim_index]
                            Sigma_ii = cov_diag[dim_index]
                            if Sigma_ii < 1e-20: mahalanobis_dists[i] = 0.0
                            else: mahalanobis_dists[i] = np.abs(mu_i - 1.0) / np.sqrt(Sigma_ii)
                        else:
                            mahalanobis_dists[i] = 0.0
                        
                    eigvals = np.linalg.eigvalsh(cov); max_eigenvalue = np.max(eigvals)
                    logger.log(itr=itr, total_evals=current_total_evals, sigma=sigma, 
                       penalized_f=penalized_fitnesses[best_idx], f=F_values[best_idx], g=g_to_log, 
                       mean_penalized_f=penalized_f_at_mean, mean_f=f_at_mean, mean_g=np.max(g_at_mean) if g_at_mean.size > 0 else 0.0,
                       v1=np.ones(dim), mean=mean, N_eff=dim, infeasible_prob=infeasible_prob, max_eigenvalue=max_eigenvalue,
                       mahalanobis_dists=mahalanobis_dists, lam=al.lam, mu=al.mu,
                       f_diff=f_diff, penalty_k_vector=penalty_k_vector)

                best_z_vectors_z = z_vectors[sorted_indices[:mu]]
                wz = (w @ best_z_vectors_z).reshape(dim, 1); mean_old = np.copy(mean)
                mean += cm * sigma * (sqrtC @ wz)
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * wz
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*itr + 1)) / chiN < 1.4 + 2/(dim+1) else 0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (sqrtC @ wz)
                rank_one_update = pc @ pc.T
                rank_mu_update_raw = np.sum([w[i] * (np.outer(best_z_vectors_z[i], best_z_vectors_z[i]) - np.eye(dim)) for i in range(mu)], axis=0)
                C_grad = sqrtC @ rank_mu_update_raw @ sqrtC.T
                C *= 1 - c1 + (1 - hsig) * c1 * cc * (2 - cc)
                C += hsig * c1 * rank_one_update + cmu * C_grad
                sigma *= np.exp(min(1.0, (cs / ds) * (np.linalg.norm(ps) / chiN - 1)))
                al.update(problem(mean.flatten()), problem.constraint(mean.flatten()))
                if hasattr(problem, 'f_opt'):
                    best_f = F_values[best_idx]; g_vals = G_values[best_idx]
                    is_feasible = np.all(g_vals <= 1e-6)
                    if is_feasible and abs(best_f - problem.f_opt) < 1e-6:
                        target_hit = True;
                        if logger: logger.close()
                        return current_total_evals
                cond_C = np.linalg.cond(C)
                if cond_C > 1e14: break
                if sigma < 1e-12: break
            remaining_budget -= evals_this_run; total_evals += evals_this_run
            if target_hit: break
            popsize_multiplier *= 2.0
    finally:
        if logger: logger.close()
    return None

def run_al_cmaes_led(problem, budget, initial_mean_generator=None, log_filepath=None):
    dim = problem.dimension; remaining_budget = budget; popsize_multiplier = 1.0; total_evals = 0
    al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5; al.chi_domega = 2**(1/(5*dim))
    is_al_initialized = False
    if log_filepath: logger = LoggerCallback(problem, log_filepath)
    else: logger = None
    try:
        while remaining_budget > 0:
            target_hit = False
            mean = (initial_mean_generator() if initial_mean_generator else np.random.uniform(-5, 5, (dim, 1))).reshape(dim, 1)
            sigma = 1.0; C = np.eye(dim); ps = np.zeros((dim, 1)); pc = np.zeros((dim, 1))
            lamb = int((4 + math.floor(3 * np.log(dim))) * popsize_multiplier)
            if remaining_budget < lamb: break
            mu = lamb // 2; wrh = np.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
            w = wrh / sum(wrh); mueff = 1 / np.sum(w**2); cm = 1.0
            evals_this_run = 0; itr = 0
            
            beta_hat = 0.01; gain_power_max = 3; gain_power_min = -2
            signal_thr = (0.106 + 0.0776*np.log(dim)) * (0.0665 + 0.947/np.sqrt(lamb))
            s_m = np.zeros((dim, 1)); s_cov = np.zeros((dim, dim)); gamma_m = np.zeros((dim, 1)); gamma_cov = np.zeros((dim, 1))
            v1 = np.ones((dim, 1)); v_sum = dim; p_v = np.ones((dim, 1)); EPS = 1e-8
            while True:
                if evals_this_run + lamb > remaining_budget: break
                itr += 1
                eigens, evecs = np.linalg.eigh(C); eigens[eigens < 0] = 0; sqrtC = evecs @ np.diag(np.sqrt(eigens)) @ evecs.T
                z_vectors = np.random.randn(lamb, dim)
                solutions = (mean.T + sigma * (sqrtC @ z_vectors.T).T)
                F_values = [problem(s) for s in solutions]; G_values = [problem.constraint(s) for s in solutions]
                if not is_al_initialized:
                    al.lam, al.mu = initialize_lagrangian_parameters(F_values, G_values, dim)
                    is_al_initialized = True
                penalized_fitnesses = [f + sum(al(g)) for f, g in zip(F_values, G_values)]
                sorted_indices = np.argsort(penalized_fitnesses); best_idx = sorted_indices[0]
                current_total_evals = total_evals + evals_this_run + lamb
                evals_this_run += lamb

                if logger:
                    f_at_mean = problem(mean.flatten()); g_at_mean = problem.constraint(mean.flatten())
                    penalized_f_at_mean = f_at_mean + sum(al(g_at_mean))
                    g_vals = G_values[best_idx]; g_to_log = np.max(g_vals) if g_vals.size > 0 else 0.0
                    N_eff = v_sum; cov = sigma**2 * C
                    violating_samples = sum(1 for g_array in G_values if np.any(g_array > 0))
                    infeasible_prob = violating_samples / lamb

                    if hasattr(problem, 'f_opt'):
                        f_diff = f_at_mean - problem.f_opt
                    else:
                        f_diff = f_at_mean
                
                    penalty_k_vector = calculate_al_penalty_terms(al.lam, al.mu, g_at_mean)
                    n_cons = len(al.lam)
                    mahalanobis_dists = np.zeros(n_cons)
                    current_mean_flat = mean.flatten()
                    cov_diag = cov.diagonal()
                    
                    for i in range(n_cons): 
                        dim_index = i + 1 
                        if dim_index < dim:
                            mu_i = current_mean_flat[dim_index]
                            Sigma_ii = cov_diag[dim_index]
                            if Sigma_ii < 1e-20: mahalanobis_dists[i] = 0.0
                            else: mahalanobis_dists[i] = np.abs(mu_i - 1.0) / np.sqrt(Sigma_ii)
                        else:
                            mahalanobis_dists[i] = 0.0
                        
                    eigvals, _ = np.linalg.eigh(cov); max_eigenvalue = np.max(eigvals)
                    logger.log(itr=itr, total_evals=current_total_evals, sigma=sigma, 
                       penalized_f=penalized_fitnesses[best_idx], f=F_values[best_idx], g=g_to_log, 
                       mean_penalized_f=penalized_f_at_mean, mean_f=f_at_mean, mean_g=np.max(g_at_mean) if g_at_mean.size > 0 else 0.0,
                       v1=v1, mean=mean, N_eff=v_sum, infeasible_prob=infeasible_prob, max_eigenvalue=max_eigenvalue,
                       mahalanobis_dists=mahalanobis_dists, lam=al.lam, mu=al.mu,
                       f_diff=f_diff, penalty_k_vector=penalty_k_vector)
                
                best_z_vectors_z = z_vectors[sorted_indices[:mu]]
                wz = (w @ best_z_vectors_z).reshape(dim, 1); mean_old = np.copy(mean)
                m_grad = sigma * (sqrtC @ wz); mean += cm * m_grad

                cs = (mueff + 2) / (v_sum + mueff + 5); ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (v_sum + 1)) - 1)
                cc = (4 + mueff / v_sum) / (v_sum + 4 + 2 * mueff / v_sum)
                alpha_cov = min(2.0, lamb / 3.0); c1 = alpha_cov / ((v_sum + 1.3)**2 + mueff); cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1/mueff) / ((v_sum + 2)**2 + alpha_cov * mueff / 2))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * wz * np.sqrt(v1)
                hsig = 1 if (np.linalg.norm(ps)**2) / (1 - (1 - cs)**(2 * itr)) < (2 + 4. / (v_sum + 1)) * np.sum(p_v) else 0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (sqrtC @ wz)
                rank_one_update = pc @ pc.T
                rank_mu_update_raw = np.sum([w[i] * (np.outer(best_z_vectors_z[i], best_z_vectors_z[i]) - np.eye(dim)) for i in range(mu)], axis=0)
                C_grad = sqrtC @ rank_mu_update_raw @ sqrtC.T
                C *= 1 - c1 + (1 - hsig) * c1 * cc * (2 - cc)
                C += hsig * c1 * rank_one_update + cmu * C_grad
                relative_velocity = (np.linalg.norm(ps)**2 / np.sum(p_v)) - 1.0
                sigma *= np.exp(min(1.0, (cs / ds) * relative_velocity))
                est_rotate = evecs.T; coef1, coef2 = 1 - beta_hat, np.sqrt(beta_hat * (2 - beta_hat))
                z_w = est_rotate @ m_grad; zz_w = est_rotate @ C_grad @ est_rotate.T
                s_m = coef1 * s_m + coef2 * z_w / (np.abs(z_w) + EPS); gamma_m = (coef1**2) * gamma_m + (coef2**2)
                s_cov = coef1 * s_cov + coef2 * zz_w / (np.abs(zz_w) + EPS); gamma_cov = (coef1**2) * gamma_cov + (coef2**2)
                signals = beta_hat / (2 - beta_hat) * np.maximum(s_m**2 / (gamma_m + EPS), np.diag(s_cov)[:,np.newaxis]**2 / (gamma_cov + EPS))
                gain1 = 10**(np.max(signals) * (gain_power_max - gain_power_min) + gain_power_min)
                def sigmoid(x, gain): return 1. / (1 + np.exp(-gain * x))
                v1 = sigmoid(signals - signal_thr, gain1) / (sigmoid(1, gain1))
                v_sum = np.sum(v1); p_v = ((1 - cs)**2) * p_v + cs * (2 - cs) * v1
                cs = (mueff + 2) / (v_sum + mueff + 5); ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (v_sum + 1)) - 1)
                cc = (4 + mueff / v_sum) / (v_sum + 4 + 2 * mueff / v_sum)
                c1 = alpha_cov / ((v_sum + 1.3)**2 + mueff); cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1/mueff) / ((v_sum + 2)**2 + alpha_cov * mueff / 2))
                al.update(problem(mean.flatten()), problem.constraint(mean.flatten()))
                
                if hasattr(problem, 'f_opt'):
                    best_f = F_values[best_idx]; g_vals = G_values[best_idx]
                    is_feasible = np.all(g_vals <= 1e-6)
                    if is_feasible and abs(best_f - problem.f_opt) < 1e-6:
                        target_hit = True;
                        if logger: logger.close()
                        return current_total_evals
                cond_C = np.linalg.cond(C)
                if cond_C > 1e14:
                    print(f"    - AL-CMA-ES-LED restarting. Reason: ConditionCov > 1e14 (is {cond_C:.1e})")
                    break
                if sigma < 1e-12:
                    print(f"    - AL-CMA-ES-LED restarting. Reason: Sigma < 1e-12 (is {sigma:.1e})")
                    break
            remaining_budget -= evals_this_run; total_evals += evals_this_run
            if target_hit: break
            popsize_multiplier *= 2.0
    finally:
        if logger: logger.close()
    return None

def run_al_cmaes_monitoring(problem, budget, initial_mean_generator=None, log_filepath=None):
    dim = problem.dimension; remaining_budget = budget; popsize_multiplier = 1.0; total_evals = 0
    al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5; al.chi_domega = 2**(1/(5*dim))
    is_al_initialized = False
    if log_filepath: logger = LoggerCallback(problem, log_filepath)
    else: logger = None
    try:
        while remaining_budget > 0:
            target_hit = False
            mean = (initial_mean_generator() if initial_mean_generator else np.random.uniform(-5, 5, (dim, 1))).reshape(dim, 1)
            sigma = 1.0; C = np.eye(dim); ps = np.zeros((dim, 1)); pc = np.zeros((dim, 1))
            lamb = int((4 + math.floor(3 * np.log(dim))) * popsize_multiplier)
            if remaining_budget < lamb: break
            mu = lamb // 2; wrh = np.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
            w = wrh / sum(wrh); mueff = 1 / np.sum(w**2); cm = 1.0
            evals_this_run = 0; itr = 0
            
            beta_hat = 0.01; gain_power_max = 3; gain_power_min = -2
            signal_thr = (0.106 + 0.0776*np.log(dim)) * (0.0665 + 0.947/np.sqrt(lamb))
            s_m = np.zeros((dim, 1)); s_cov = np.zeros((dim, dim)); gamma_m = np.zeros((dim, 1)); gamma_cov = np.zeros((dim, 1))
            v1 = np.ones((dim, 1)); v_sum = dim; p_v = np.ones((dim, 1)); EPS = 1e-8
            
            cs = (mueff + 2) / (dim + mueff + 5); ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            alpha_cov = min(2.0, lamb / 3.0); c1 = alpha_cov / ((dim + 1.3)**2 + mueff); cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + alpha_cov * mueff / 2))
            chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

            while True:
                if evals_this_run + lamb > remaining_budget: break
                itr += 1
                eigens, evecs = np.linalg.eigh(C); eigens[eigens < 0] = 0; sqrtC = evecs @ np.diag(np.sqrt(eigens)) @ evecs.T
                z_vectors = np.random.randn(lamb, dim)
                solutions = (mean.T + sigma * (sqrtC @ z_vectors.T).T)
                F_values = [problem(s) for s in solutions]; G_values = [problem.constraint(s) for s in solutions]
                if not is_al_initialized:
                    al.lam, al.mu = initialize_lagrangian_parameters(F_values, G_values, dim)
                    is_al_initialized = True
                penalized_fitnesses = [f + sum(al(g)) for f, g in zip(F_values, G_values)]
                sorted_indices = np.argsort(penalized_fitnesses); best_idx = sorted_indices[0]
                current_total_evals = total_evals + evals_this_run + lamb
                evals_this_run += lamb

                if logger:
                    f_at_mean = problem(mean.flatten()); g_at_mean = problem.constraint(mean.flatten())
                    penalized_f_at_mean = f_at_mean + sum(al(g_at_mean))
                    g_vals = G_values[best_idx]; g_to_log = np.max(g_vals) if g_vals.size > 0 else 0.0
                    N_eff = v_sum; cov = sigma**2 * C 
                    violating_samples = sum(1 for g_array in G_values if np.any(g_array > 0))
                    infeasible_prob = violating_samples / lamb
                    if hasattr(problem, 'f_opt'):
                        f_diff = f_at_mean - problem.f_opt
                    else:
                        f_diff = f_at_mean
                    penalty_k_vector = calculate_al_penalty_terms(al.lam, al.mu, g_at_mean)
                    
                    n_cons = len(al.lam)
                    mahalanobis_dists = np.zeros(n_cons)
                    current_mean_flat = mean.flatten()
                    cov_diag = cov.diagonal()
                    for i in range(n_cons): 
                        dim_index = i + 1 
                        if dim_index < dim:
                            mu_i = current_mean_flat[dim_index]
                            Sigma_ii = cov_diag[dim_index]
                            if Sigma_ii < 1e-20: mahalanobis_dists[i] = 0.0
                            else: mahalanobis_dists[i] = np.abs(mu_i - 1.0) / np.sqrt(Sigma_ii)
                        else:
                            mahalanobis_dists[i] = 0.0

                    eigvals, _ = np.linalg.eigh(cov); max_eigenvalue = np.max(eigvals)
                    
                    logger.log(itr=itr, total_evals=current_total_evals, sigma=sigma, 
                       penalized_f=penalized_fitnesses[best_idx], f=F_values[best_idx], g=g_to_log, 
                       mean_penalized_f=penalized_f_at_mean, mean_f=f_at_mean, mean_g=np.max(g_at_mean) if g_at_mean.size > 0 else 0.0,
                       v1=v1, mean=mean, N_eff=v_sum, infeasible_prob=infeasible_prob, max_eigenvalue=max_eigenvalue,
                       mahalanobis_dists=mahalanobis_dists, lam=al.lam, mu=al.mu,
                       f_diff=f_diff, penalty_k_vector=penalty_k_vector)
                
                best_z_vectors_z = z_vectors[sorted_indices[:mu]]
                wz = (w @ best_z_vectors_z).reshape(dim, 1); mean_old = np.copy(mean)
                m_grad = sigma * (sqrtC @ wz); mean += cm * m_grad

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * wz 
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*itr + 1)) / chiN < 1.4 + 2/(dim+1) else 0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (sqrtC @ wz)
                rank_one_update = pc @ pc.T
                rank_mu_update_raw = np.sum([w[i] * (np.outer(best_z_vectors_z[i], best_z_vectors_z[i]) - np.eye(dim)) for i in range(mu)], axis=0)
                C_grad = sqrtC @ rank_mu_update_raw @ sqrtC.T
                C *= 1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)
                C += hsig * c1 * rank_one_update + cmu * C_grad
                
                sigma *= np.exp(min(1.0, (cs / ds) * (np.linalg.norm(ps) / chiN - 1)))

                est_rotate = evecs.T; coef1, coef2 = 1 - beta_hat, np.sqrt(beta_hat * (2 - beta_hat))
                z_w = est_rotate @ m_grad; zz_w = est_rotate @ C_grad @ est_rotate.T
                s_m = coef1 * s_m + coef2 * z_w / (np.abs(z_w) + EPS); gamma_m = (coef1**2) * gamma_m + (coef2**2)
                s_cov = coef1 * s_cov + coef2 * zz_w / (np.abs(zz_w) + EPS); gamma_cov = (coef1**2) * gamma_cov + (coef2**2)
                signals = beta_hat / (2 - beta_hat) * np.maximum(s_m**2 / (gamma_m + EPS), np.diag(s_cov)[:,np.newaxis]**2 / (gamma_cov + EPS))
                gain1 = 10**(np.max(signals) * (gain_power_max - gain_power_min) + gain_power_min)
                def sigmoid(x, gain): return 1. / (1 + np.exp(-gain * x))
                v1 = sigmoid(signals - signal_thr, gain1) / (sigmoid(1, gain1) + EPS)
                v_sum = np.sum(v1); p_v = ((1 - cs)**2) * p_v + cs * (2 - cs) * v1

                al.update(problem(mean.flatten()), problem.constraint(mean.flatten()))

                if hasattr(problem, 'f_opt'):
                    best_f = F_values[best_idx]; g_vals = G_values[best_idx]
                    is_feasible = np.all(g_vals <= 1e-6)
                    if is_feasible and abs(best_f - problem.f_opt) < 1e-6:
                        target_hit = True;
                        if logger: logger.close()
                        return current_total_evals
                cond_C = np.linalg.cond(C)
                if cond_C > 1e14: break
                if sigma < 1e-12: break
            remaining_budget -= evals_this_run; total_evals += evals_this_run
            if target_hit: break
            popsize_multiplier *= 2.0
    finally:
        if logger: logger.close()
    return None