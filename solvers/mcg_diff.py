from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from mcg_diff.particle_filter import mcg_diff
    from mcg_diff.sgm import ScoreModel
    from mcg_diff.utils import NetReparametrized, get_optimal_timesteps_from_singular_values
    from functools import partial
    import torch


def build_extended_svd(A):
    U, d, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = torch.ones_like(V[0])
    coordinate_mask[len(d):] = 0
    return U, d, coordinate_mask, V


class Solver(BaseSolver):
    name = 'Python-MCG_Diff'  # proximal gradient, optionally accelerated

    # Any parameter defined here is accessible as an attribute of the solver.
    parameters = {"ddim_params": [(100, .6)], "n_particles": [32, 128]}

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
        u, diag, coordinate_mask, v = build_extended_svd(forward_operator)
        score_model = ScoreModel(NetReparametrized(base_score_module=lambda x, t: -((1 - alphas_cumprod[t])**.5)*score_network(x, alphas_cumprod[t]),
                                                   orthogonal_transformation=v),
                                 alphas_cumprod=alphas_cumprod,
                                 device='cpu')

        n_steps, eta = self.ddim_params
        adapted_timesteps = get_optimal_timesteps_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                                       singular_value=diag,
                                                                       n_timesteps=n_steps,
                                                                       var=observation_noise,
                                                                       mode='else')
        def mcg_diff_fun(initial_samples):
            samples, log_weights = mcg_diff(
                initial_particles=initial_samples,
                observation=(u.T @ observation),
                score_model=score_model,
                likelihood_diagonal=diag,
                coordinates_mask=coordinate_mask.bool(),
                var_observation=observation_noise,
                timesteps=adapted_timesteps,
                eta=eta,
                gaussian_var=1e-6,
            )
            return v.T @ samples[torch.distributions.Categorical(logits=log_weights, validate_args=False).sample(sample_shape=(1,))][0]
        self.sampler = mcg_diff_fun
        self.dim_y, self.dim_x = forward_operator.shape
        self.n_samples = n_samples

    def run(self, n_iter):
        initial_samples = torch.randn(size=(self.n_samples, self.n_particles, self.dim_x))
        self.samples = torch.func.vmap(self.sampler, in_dims=(0,), randomness='different')(initial_samples)

    # Return the solution estimate computed.
    def get_result(self):
        return {'samples': self.samples}