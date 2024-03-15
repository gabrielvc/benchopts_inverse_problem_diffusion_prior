from benchopt import BaseSolver, safe_import_context
from abc import ABC, abstractmethod

with safe_import_context() as import_ctx:
    import torch


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


class Solver(BaseSolver):
    name = 'Python-DPS'  # proximal gradient, optionally accelerated

    # Any parameter defined here is accessible as an attribute of the solver.
    parameters = {"ddim_params": [(100, .6)], "zeta":[1.]}

    # Store the information to compute the objective. The parameters of this
    # function are the keys of the dictionary obtained when calling
    # ``Objective.get_objective``.
    def set_objective(self,
                      observation,
                      observation_noise,
                      forward_operator,
                      score_network,
                      alphas_cumprod,
                      n_samples,
                      betas):
        self.dim_y, self.dim_x = forward_operator.shape
        self.n_samples = n_samples
        n_steps, self.eta = self.ddim_params
        self.timesteps = torch.linspace(0, 999, n_steps).long().tolist()
        self.alphas_cumprod = alphas_cumprod

        def dps_grad_fun(sample, alpha_t):
            score = score_network(sample, alpha_t)
            pred_x0 = (1 / (alpha_t**.5))*(sample + (1 - alpha_t) * score)
            residue = torch.linalg.norm(forward_operator @ pred_x0 - observation)
            return residue**2, pred_x0

        self.dps_fn = torch.func.vmap(torch.func.grad_and_value(dps_grad_fun, has_aux=True), in_dims=(0, None))


    def run(self, n_iter):
        n_steps, eta = self.ddim_params
        samples = torch.randn((self.n_samples, self.dim_x))
        for i, (t, t_prev) in enumerate(zip(self.timesteps[1:][::-1], self.timesteps[:-1][::-1])):
            alpha_t, alpha_t_prev = self.alphas_cumprod[t], self.alphas_cumprod[t_prev]
            grad_residue, (residue_sq, pred_x0) = self.dps_fn(samples, alpha_t)
            z = torch.randn_like(samples)
            coeff_sample = (alpha_t ** .5) * (1 - alpha_t_prev) / (1 - alpha_t)
            coeff_pred = (alpha_t ** .5) * (1 - alpha_t / alpha_t_prev) / (1 - alpha_t)
            coeff_lik_score = self.zeta / residue_sq**.5
            noise_std = eta * (((1 - alpha_t_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_prev)) ** .5

            samples = coeff_sample * samples + coeff_pred * pred_x0 - coeff_lik_score[:, None]*grad_residue + noise_std * z
        self.samples = samples

    # Return the solution estimate computed.
    def get_result(self):
        return {'samples': self.samples}
