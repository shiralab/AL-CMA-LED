import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

import cma
import numpy as np
import math
import os
from cma.evolution_strategy import CMAEvolutionStrategy
from cma.constraints_handler import AugmentedLagrangian

# ラグランジュ関数のパラメータの初期化
def initialize_lagrangian_parameters(F_values, G_values, dimension):
    F_arr = np.array(F_values, dtype=float)
    if np.all(np.isnan(F_arr)):
        delta_f = 1.0 
    else:
        delta_f = np.nanpercentile(F_arr, 75) - np.nanpercentile(F_arr, 25)
    
    if len(G_values) == 0 or G_values[0] is None: 
        return np.array([]), np.array([])
    
    n_constraints = len(G_values[0])
    lam = np.zeros(n_constraints)
    mu = np.zeros(n_constraints)
    
    for k in range(n_constraints):
        g_k_values = [g[k] for g in G_values if g is not None]
        if not g_k_values: 
            lam[k] = 1.0; mu[k] = 1.0
            continue
            
        g_k_arr = np.array(g_k_values, dtype=float)
        

        if np.all(np.isnan(g_k_arr)):
            delta_g_k = 0.0
        else:
            delta_g_k = np.nanpercentile(g_k_arr, 75) - np.nanpercentile(g_k_arr, 25)
            
        if delta_g_k > 1e-9 and not np.isnan(delta_f):
            val_lam = delta_f / (dimension * delta_g_k + 1e-11 * (delta_f + 1))
            lam[k] = val_lam
            
            g_k_sq_values = g_k_arr**2
            if np.all(np.isnan(g_k_sq_values)):
                delta_g_k_sq = 0.0
            else:
                delta_g_k_sq = np.nanpercentile(g_k_sq_values, 75) - np.nanpercentile(g_k_sq_values, 25)
                
            val_mu = (2 * delta_f) / (5 * dimension * (delta_g_k + 1e-6 * delta_g_k_sq + 1e-11 * (delta_f + 1)))
            mu[k] = val_mu
        else:
            lam[k] = 1.0; mu[k] = 1.0
            

    lam = np.nan_to_num(lam, nan=1.0, posinf=1.0, neginf=1.0)
    mu = np.nan_to_num(mu, nan=1.0, posinf=1.0, neginf=1.0)
    lam = np.abs(lam)
    mu = np.abs(mu)
    
    return lam, mu

# al-cma-es
def run_al_cmaes(problem, budget):
    constrained_fitness_function = cma.ConstrainedFitnessAL(fun=problem, constraints=problem.constraint, dimension=problem.dimension)
    al_handler = constrained_fitness_function.al
    al_handler.set_algorithm(1); al_handler.dgamma = 5; al_handler.chi_domega = 2**(1/(5*problem.dimension))
    al_handler.k1 = 3; al_handler.k2 = 5
    opts = cma.CMAOptions()
    opts['bounds'] = [-5, 5]; opts['maxfevals'] = budget
    opts['tolstagnation'] = 0; opts['verbose'] = -9
    cma.fmin2(
        objective_function=constrained_fitness_function,
        x0=lambda: np.random.uniform(-5, 5, problem.dimension),
        sigma0=1.0, options=opts, callback=constrained_fitness_function.update,
        restarts=999, incpopsize=2
    )

# al-cma-es-led(提案手法)
def run_al_cmaes_led(problem, budget):
    dim = problem.dimension; remaining_budget = budget; popsize_multiplier = 1.0
    al = None
    is_al_initialized = False

    tol_fun = 1e-11
    tol_fun_hist = 1e-12

    while remaining_budget > 0:
        is_al_initialized = False
        mean = np.random.uniform(-5, 5, (dim, 1))
        sigma = 1.0; C = np.eye(dim); ps = np.zeros((dim, 1)); pc = np.zeros((dim, 1))
        lamb = int((4 + math.floor(3 * np.log(dim))) * popsize_multiplier)
        if remaining_budget < lamb: break
        mu = lamb // 2; wrh = np.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
        w = wrh / sum(wrh); mueff = 1 / np.sum(w**2); cm = 1.0
        evals_this_run = 0; itr = 0
        al = None

        fitness_history = []

        beta_hat_init = 0.01; beta_hat = beta_hat_init
        gain_power_max = 3; gain_power_min = -2
        signal_thr = (0.106 + 0.0776*np.log(dim)) * (0.0665 + 0.947/np.sqrt(lamb))
        s_m = np.zeros((dim, 1)); s_cov = np.zeros((dim, dim))
        gamma_m = np.zeros((dim, 1)); gamma_cov = np.zeros((dim, 1))
        v1 = np.ones((dim, 1)); v_sum = dim; p_v = np.ones((dim, 1))
        EPS = 1e-8; I = np.eye(dim, dtype=float)
        
        while True:
            if evals_this_run + lamb > remaining_budget: break
            itr += 1
            eigens, evecs = np.linalg.eigh(C); eigens[eigens < 0] = 0
            sqrtC = evecs @ np.diag(np.sqrt(eigens)) @ evecs.T
            z_vectors = np.random.randn(lamb, dim)
            solutions = (mean.T + sigma * (sqrtC @ z_vectors.T).T)
            F_values = [problem(s) for s in solutions]
            G_values = [problem.constraint(s) for s in solutions]
            evals_this_run += lamb
            if not is_al_initialized:
                if al is None:
                    if G_values and G_values[0] is not None:
                        n_constraints = len(G_values[0])
                    else:
                        n_constraints = 0
                    
                    al = AugmentedLagrangian(n_constraints)
                    al.set_algorithm(1)
                    al.dgamma = 5
                    al.chi_domega = 2**(1/(5*dim)) 
                
                lam_init, mu_init = initialize_lagrangian_parameters(F_values, G_values, dim)
                al.lam = lam_init
                al.mu = mu_init
                is_al_initialized = True


            penalized_fitnesses = []
            for f, g in zip(F_values, G_values):
                p_val = sum(al(g))
                if np.isnan(p_val): p_val = 1e10 
                penalized_fitnesses.append(f + p_val)
                
            sorted_indices = np.argsort(penalized_fitnesses)

            best_z_vectors = z_vectors[sorted_indices[:mu]]
            
            mean_old = np.copy(mean)
            wz = (w @ best_z_vectors).reshape(dim, 1)
            m_grad = sigma * (sqrtC @ wz)
            mean += cm * m_grad
            cs = (mueff + 2.0) / (v_sum + mueff + 5.0)
            ds = 1.0 + cs + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (v_sum + 1.0)) - 1.0)
            cc = (4.0 + mueff / v_sum) / (v_sum + 4.0 + 2.0 * mueff / v_sum)
            alpha_cov = min(2.0, lamb / 3.0)
            c1 = alpha_cov / (math.pow(v_sum + 1.3, 2) + mueff)
            cmu = min(1.0 - c1, alpha_cov * (mueff - 2.0 + 1.0 / mueff) / ((v_sum + 2.0)**2 + alpha_cov * mueff / 2.0))
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * wz * np.sqrt(v1)
            hsig = 1 if (np.linalg.norm(ps)**2) / (1 - (1 - cs)**(2 * (itr))) < (2.0 + 4.0 / (v_sum + 1)) * (np.sum(p_v)) else 0
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (sqrtC @ wz)
            rone = pc @ pc.T
            zrmu = np.sum([w[i] * (np.outer(best_z_vectors[i], best_z_vectors[i]) - I) for i in range(mu)], axis=0)
            C_grad = sqrtC @ zrmu @ sqrtC.T
            C *= 1.0 - c1 + (1.0 - hsig) * c1 * cc * (2.0 - cc)
            C += hsig * c1 * rone + cmu * C_grad
            relative_velocity = (np.linalg.norm(ps)**2 / np.sum(p_v)) - 1.0
            sigma *= math.exp(min(1.0, (cs / ds) * relative_velocity))
            est_rotate = evecs.T
            coef1, coef2 = 1 - beta_hat, np.sqrt(beta_hat * (2 - beta_hat))
            z_w = est_rotate @ m_grad; zz_w = est_rotate @ C_grad @ est_rotate.T
            s_m = coef1 * s_m + coef2 * z_w / (np.abs(z_w) + EPS)
            gamma_m = (coef1**2) * gamma_m + (coef2**2) * np.ones((dim, 1))
            s_cov = coef1 * s_cov + coef2 * zz_w / (np.abs(zz_w) + EPS)
            gamma_cov = (coef1**2) * gamma_cov + (coef2**2) * np.ones((dim, 1))
            signals = beta_hat / (2 - beta_hat) * np.maximum(s_m**2 / (gamma_m + EPS), np.diag(s_cov)[:,np.newaxis]**2 / (gamma_cov + EPS))
            max_s1 = np.max(signals)
            gain_power1 = max_s1 * (gain_power_max - gain_power_min) + gain_power_min
            gain1 = 10**gain_power1
            
            def sigmoid(x, gain): return 1. / (1 + np.exp(-gain * x))
            
            v1 = sigmoid(signals - signal_thr, gain1) / (sigmoid(1, gain1) + EPS)
            v_sum = np.sum(v1)
            p_v = ((1 - cs)**2) * p_v + cs * (2 - cs) * v1
            
            c1 = alpha_cov / (math.pow(v_sum + 1.3, 2) + mueff)
            cmu = min(1.0 - c1, alpha_cov * (mueff - 2.0 + 1.0 / mueff) / ((v_sum + 2.0)**2 + alpha_cov * mueff / 2.0))
            cc = (4.0 + mueff / v_sum) / (v_sum + 4.0 + 2.0 * mueff / v_sum)
            cs = (mueff + 2.0) / (v_sum + mueff + 5.0)
            ds = 1.0 + cs + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (v_sum + 1.0)) - 1.0)
            
            try:
                raw_g = problem.constraint(mean.flatten())
                if raw_g is None:
                    safe_g = None
                else:
                    safe_g = np.clip(raw_g, -1e10, 1e10)
                
                f_mean = problem(mean.flatten()) 
                
                al.update(f_mean, safe_g)

            except (AssertionError, RuntimeWarning, ValueError) as e:
                al = None
                is_al_initialized = False


            if problem.final_target_hit: break
            if np.linalg.norm(mean - mean_old) < 1e-11 * sigma: break
            if np.linalg.cond(C) > 1e14: break

            
        remaining_budget -= evals_this_run
        if problem.final_target_hit: break
        popsize_multiplier *= 2.0

class MonitoringLogger:
    def __init__(self, problem, log_dir="monitoring_data"):
        os.makedirs(log_dir, exist_ok=True)
        clean_id = problem.id.replace(" ", "_").replace(":", "")
        self.filepath = os.path.join(log_dir, f"{clean_id}.csv")
        self.f = open(self.filepath, 'w')
        self.header_written = False
        self.dim = problem.dimension

    def log(self, evals, mean_f, mean_penalized_f, sigma, lam, mu, v1, v_sum, cond_C, mean):
        if not self.header_written:
            n_lam = len(lam)
            lam_cols = ",".join([f"lam_{i}" for i in range(n_lam)])
            mu_cols = ",".join([f"mu_{i}" for i in range(n_lam)])
            v1_cols = ",".join([f"v1_{i}" for i in range(self.dim)])
            mean_cols = ",".join([f"mean_{i}" for i in range(self.dim)])
            
            self.f.write(f"evals,mean_f,mean_penalized_f,sigma,v_sum,cond_C,{lam_cols},{mu_cols},{v1_cols},{mean_cols}\n")
            self.header_written = True

        lam_str = ",".join(map(str, lam.flatten()))
        mu_str = ",".join(map(str, mu.flatten()))
        
        v1_str = ",".join(map(str, v1.flatten()))
        mean_str = ",".join(map(str, mean.flatten()))
        
        self.f.write(f"{evals},{mean_f},{mean_penalized_f},{sigma},{v_sum},{cond_C},{lam_str},{mu_str},{v1_str},{mean_str}\n")
        self.f.flush()

    def close(self):
        self.f.close()


# 以下各手法のログ記録用コード

def run_al_cmaes_monitoring(problem, budget):
    dim = problem.dimension; remaining_budget = budget; popsize_multiplier = 1.0
    al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5
    al.chi_domega = 2**(1/(5*dim))
    is_al_initialized = False

    logger = MonitoringLogger(problem)

    try:
        while remaining_budget > 0:
            is_al_initialized = False
            mean = np.random.uniform(-5, 5, (dim, 1))
            sigma = 1.0; C = np.eye(dim); ps = np.zeros((dim, 1)); pc = np.zeros((dim, 1))
            lamb = int((4 + math.floor(3 * np.log(dim))) * popsize_multiplier)
            if remaining_budget < lamb: break
            mu = lamb // 2; wrh = np.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
            w = wrh / sum(wrh); mueff = 1 / np.sum(w**2); cm = 1.0
            evals_this_run = 0; itr = 0
            al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5
            al.chi_domega = 2**(1/(5*dim))
            is_al_initialized = False
            

            cs = (mueff + 2) / (dim + mueff + 5); ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            alpha_cov = min(2.0, lamb / 3.0); c1 = alpha_cov / ((dim + 1.3)**2 + mueff); cmu = min(1 - c1, alpha_cov * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + alpha_cov * mueff / 2))
            chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

            beta_hat = 0.01; gain_power_max = 3; gain_power_min = -2
            signal_thr = (0.106 + 0.0776*np.log(dim)) * (0.0665 + 0.947/np.sqrt(lamb))
            s_m = np.zeros((dim, 1)); s_cov = np.zeros((dim, dim))
            gamma_m = np.zeros((dim, 1)); gamma_cov = np.zeros((dim, 1))
            v1 = np.ones((dim, 1)); EPS = 1e-8; I = np.eye(dim, dtype=float)
            
            while True:
                if evals_this_run + lamb > remaining_budget: break
                itr += 1
                eigens, evecs = np.linalg.eigh(C); eigens[eigens < 0] = 0
                sqrtC = evecs @ np.diag(np.sqrt(eigens)) @ evecs.T
                z_vectors = np.random.randn(lamb, dim)
                solutions = (mean.T + sigma * (sqrtC @ z_vectors.T).T)
                
                F_values = [problem(s) for s in solutions]
                G_values = [problem.constraint(s) for s in solutions]
                evals_this_run += lamb
                
                if not is_al_initialized:
                    al.lam, al.mu = initialize_lagrangian_parameters(F_values, G_values, dim)
                    is_al_initialized = True
                penalized_fitnesses = [f + sum(al(g)) for f, g in zip(F_values, G_values)]
                sorted_indices = np.argsort(penalized_fitnesses)
                best_z_vectors = z_vectors[sorted_indices[:mu]]
                
                mean_old = np.copy(mean)
                wz = (w @ best_z_vectors).reshape(dim, 1)
                m_grad = sigma * (sqrtC @ wz)
                mean += cm * m_grad

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * wz
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*itr + 1)) / chiN < 1.4 + 2/(dim+1) else 0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (sqrtC @ wz)
                rone = pc @ pc.T
                zrmu = np.sum([w[i] * (np.outer(best_z_vectors[i], best_z_vectors[i]) - I) for i in range(mu)], axis=0)
                C_grad = sqrtC @ zrmu @ sqrtC.T
                
                C *= 1.0 - c1  + (1.0 - hsig) * c1 * cc * (2.0 - cc)
                C += hsig * c1 * rone + cmu * C_grad
                
                sigma *= math.exp(min(1.0, (cs / ds) * (np.linalg.norm(ps) / chiN - 1)))

                est_rotate = evecs.T; coef1, coef2 = 1 - beta_hat, np.sqrt(beta_hat * (2 - beta_hat))
                z_w = est_rotate @ m_grad; zz_w = est_rotate @ C_grad @ est_rotate.T
                s_m = coef1 * s_m + coef2 * z_w / (np.abs(z_w) + EPS)
                gamma_m = (coef1**2) * gamma_m + (coef2**2) * np.ones((dim, 1))
                s_cov = coef1 * s_cov + coef2 * zz_w / (np.abs(zz_w) + EPS)
                gamma_cov = (coef1**2) * gamma_cov + (coef2**2) * np.ones((dim, 1))
                signals = beta_hat / (2 - beta_hat) * np.maximum(s_m**2 / (gamma_m + EPS), np.diag(s_cov)[:,np.newaxis]**2 / (gamma_cov + EPS))
                gain1 = 10**(np.max(signals) * (gain_power_max - gain_power_min) + gain_power_min)
                
                v1 = 1. / (1 + np.exp(-gain1 * (signals - signal_thr))) / (1. / (1 + np.exp(-gain1)) + EPS)
                v_sum = np.sum(v1)
                
                mean_f = problem(mean.flatten())
                cond_C = np.linalg.cond(C)
                logger.log(problem.evaluations, mean_f, mean_f, sigma, al.lam, al.mu, v1, v_sum, cond_C, mean)

                try:
                    raw_g = problem.constraint(mean.flatten())
                    if raw_g is None:
                        safe_g = None
                    else:
                        safe_g = np.clip(raw_g, -1e10, 1e10)
                
                    f_mean = problem(mean.flatten()) 
                
                    al.update(f_mean, safe_g)

                except (AssertionError, RuntimeWarning, ValueError) as e:
                    al = None
                    is_al_initialized = False
            
                if problem.final_target_hit: break
                if np.linalg.norm(mean - mean_old) < 1e-11 * sigma: break
                if np.linalg.cond(C) > 1e14: break
                if sigma < 1e-12: break
            
            remaining_budget -= evals_this_run
            if problem.final_target_hit: break
            popsize_multiplier *= 2.0
            print("リスタート")
    finally:
        logger.close()

def run_al_cmaes_led_monitoring(problem, budget):
    dim = problem.dimension; remaining_budget = budget; popsize_multiplier = 1.0
    al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5
    al.chi_domega = 2**(1/(5*dim))
    is_al_initialized = False

    tol_fun = 1e-11       
    tol_fun_hist = 1e-12  

    logger = MonitoringLogger(problem, log_dir="led_v1_data")

    try: 
        while remaining_budget > 0:
            is_al_initialized = False
            mean = np.random.uniform(-5, 5, (dim, 1))
            sigma = 1.0; C = np.eye(dim); ps = np.zeros((dim, 1)); pc = np.zeros((dim, 1))
            lamb = int((4 + math.floor(3 * np.log(dim))) * popsize_multiplier)
            if remaining_budget < lamb: break
            mu = lamb // 2; wrh = np.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
            w = wrh / sum(wrh); mueff = 1 / np.sum(w**2); cm = 1.0
            evals_this_run = 0; itr = 0
            al = AugmentedLagrangian(dim); al.set_algorithm(1); al.dgamma = 5
            al.chi_domega = 2**(1/(5*dim))
            is_al_initialized = False

            fitness_history = []

            beta_hat_init = 0.01; beta_hat = beta_hat_init
            gain_power_max = 3; gain_power_min = -2
            signal_thr = (0.106 + 0.0776*np.log(dim)) * (0.0665 + 0.947/np.sqrt(lamb))
            s_m = np.zeros((dim, 1)); s_cov = np.zeros((dim, dim))
            gamma_m = np.zeros((dim, 1)); gamma_cov = np.zeros((dim, 1))
            v1 = np.ones((dim, 1)); v_sum = dim; p_v = np.ones((dim, 1))
            EPS = 1e-8; I = np.eye(dim, dtype=float)
            
            while True:
                if evals_this_run + lamb > remaining_budget: break
                itr += 1
                eigens, evecs = np.linalg.eigh(C); eigens[eigens < 0] = 0
                sqrtC = evecs @ np.diag(np.sqrt(eigens)) @ evecs.T
                z_vectors = np.random.randn(lamb, dim)
                solutions = (mean.T + sigma * (sqrtC @ z_vectors.T).T)
                F_values = [problem(s) for s in solutions]
                G_values = [problem.constraint(s) for s in solutions]
                evals_this_run += lamb
                if not is_al_initialized:
                    al.lam, al.mu = initialize_lagrangian_parameters(F_values, G_values, dim)
                    is_al_initialized = True
                penalized_fitnesses = [f + sum(al(g)) for f, g in zip(F_values, G_values)]
                sorted_indices = np.argsort(penalized_fitnesses)

                best_z_vectors = z_vectors[sorted_indices[:mu]]
                
                mean_old = np.copy(mean)
                wz = (w @ best_z_vectors).reshape(dim, 1)
                m_grad = sigma * (sqrtC @ wz)
                mean += cm * m_grad
                cs = (mueff + 2.0) / (v_sum + mueff + 5.0)
                ds = 1.0 + cs + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (v_sum + 1.0)) - 1.0)
                cc = (4.0 + mueff / v_sum) / (v_sum + 4.0 + 2.0 * mueff / v_sum)
                alpha_cov = min(2.0, lamb / 3.0)
                c1 = alpha_cov / (math.pow(v_sum + 1.3, 2) + mueff)
                cmu = min(1.0 - c1, alpha_cov * (mueff - 2.0 + 1.0 / mueff) / ((v_sum + 2.0)**2 + alpha_cov * mueff / 2.0))
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * wz * np.sqrt(v1)
                hsig = 1 if (np.linalg.norm(ps)**2) / (1 - (1 - cs)**(2 * (itr + 1))) < (2.0 + 4.0 / (v_sum + 1)) * (np.sum(p_v)) else 0 
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (sqrtC @ wz)
                rone = pc @ pc.T
                zrmu = np.sum([w[i] * (np.outer(best_z_vectors[i], best_z_vectors[i]) - I) for i in range(mu)], axis=0)
                C_grad = sqrtC @ zrmu @ sqrtC.T
                C *= 1.0 - c1 + (1.0 - hsig) * c1 * cc * (2.0 - cc) 
                C += hsig * c1 * rone + cmu * C_grad
                relative_velocity = (np.linalg.norm(ps)**2 / np.sum(p_v)) - 1.0
                sigma *= math.exp(min(1.0, (cs / ds) * relative_velocity))
                est_rotate = evecs.T
                coef1, coef2 = 1 - beta_hat, np.sqrt(beta_hat * (2 - beta_hat))
                z_w = est_rotate @ m_grad; zz_w = est_rotate @ C_grad @ est_rotate.T
                s_m = coef1 * s_m + coef2 * z_w / (np.abs(z_w) + EPS)
                gamma_m = (coef1**2) * gamma_m + (coef2**2) * np.ones((dim, 1))
                s_cov = coef1 * s_cov + coef2 * zz_w / (np.abs(zz_w) + EPS)
                gamma_cov = (coef1**2) * gamma_cov + (coef2**2) * np.ones((dim, 1))
                signals = beta_hat / (2 - beta_hat) * np.maximum(s_m**2 / (gamma_m + EPS), np.diag(s_cov)[:,np.newaxis]**2 / (gamma_cov + EPS))
                max_s1 = np.max(signals)
                gain_power1 = max_s1 * (gain_power_max - gain_power_min) + gain_power_min
                gain1 = 10**gain_power1
                
                def sigmoid(x, gain): return 1. / (1 + np.exp(-gain * x))
                
                v1 = sigmoid(signals - signal_thr, gain1) / (sigmoid(1, gain1))
                v_sum = np.sum(v1)
                
                mean_f = problem(mean.flatten())
                cond_C = np.linalg.cond(C)
                logger.log(problem.evaluations, mean_f, mean_f, sigma, al.lam, al.mu, v1, v_sum, cond_C, mean)

                p_v = ((1 - cs)**2) * p_v + cs * (2 - cs) * v1
                
                c1 = alpha_cov / (math.pow(v_sum + 1.3, 2) + mueff)
                cmu = min(1.0 - c1, alpha_cov * (mueff - 2.0 + 1.0 / mueff) / ((v_sum + 2.0)**2 + alpha_cov * mueff / 2.0))
                cc = (4.0 + mueff / v_sum) / (v_sum + 4.0 + 2.0 * mueff / v_sum)
                cs = (mueff + 2.0) / (v_sum + mueff + 5.0)
                ds = 1.0 + cs + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (v_sum + 1.0)) - 1.0)
                
                try:
                    raw_g = problem.constraint(mean.flatten())
                    if raw_g is None:
                        safe_g = None
                    else:
                        safe_g = np.clip(raw_g, -1e10, 1e10)
                
                    f_mean = problem(mean.flatten()) 
                
                    al.update(f_mean, safe_g)

                except (AssertionError, RuntimeWarning, ValueError) as e:
                    al = None
                    is_al_initialized = False

                if problem.final_target_hit: break
                if np.linalg.norm(mean - mean_old) < 1e-11 * sigma: break
                if np.linalg.cond(C) > 1e14: break
                
            remaining_budget -= evals_this_run
            if problem.final_target_hit: 
                print("fini")
                break
            popsize_multiplier *= 2.0
            print("リスタート:")
            print(remaining_budget)
            
    finally:
        logger.close()