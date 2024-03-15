from benchopt import BaseObjective, safe_import_context

# All packages other than benchopt should be imported in this context.
# - This allows to list solvers even when a package is not installed,
#   in particular for listing dependencies to install.
# - This allows to skip imports when listing solvers and datasets
#   for auto completion.
with safe_import_context() as import_ctx:
    import numpy as np
    from ot.sliced import max_sliced_wasserstein_distance
    import matplotlib.pyplot as plt


class Objective(BaseObjective):
    # Name of the Objective function
    name = "Diffusion posterior sampling for Linear inverse problems with Gaussian noise"
    color_algo = 'red'
    color_reference_samples = 'blue'
    # parametrization of the objective with various regularization parameters.
    # All parameters `p` defined here will be accessible as `self.p`.
    n_samples = 1_000

    def get_one_result(self):
        "Return one solution for which the objective can be evaluated."
        return np.zeros((self.n_samples, self.reference_samples.shape[-1]))

    def set_data(self,
                 observation,
                 forward_operator,
                 observation_noise,
                 score_network,
                 alphas_cumprod,
                 betas,
                 reference_samples):
        """Set the data from a Dataset to compute the objective.

        The argument are the keys in the data dictionary returned by
        get_data.
        """
        self.observation = observation
        self.forward_operator = forward_operator
        self.observation_noise = observation_noise
        self.reference_samples = reference_samples
        self.alphas_cumprod = alphas_cumprod
        self.score_network = score_network
        self.betas = betas

    def evaluate_result(self, samples):
        """Compute the objective value given the output of a solver.

        The arguments are the keys in the result dictionary returned
        by ``Solver.get_result``.
        """
        # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # fig.subplots_adjust(left=0, right=1,
        #                     bottom=0, top=1)
        # # ax.scatter(*prior_to_plot.T, edgecolors="black", alpha=.2, rasterized=True, c=color_prior)
        # ax.scatter(*self.reference_samples[:, :2].T, edgecolors="black", alpha=.2, c=self.color_reference_samples, rasterized=True)
        # ax.scatter(*samples[:, :2].T, alpha=.2, edgecolors="black", c=self.color_algo, rasterized=True)
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(-20, 20)
        # fig.show()
        # plt.close(fig)
        sliced_wasserstein_distance = max_sliced_wasserstein_distance(X_s=self.reference_samples,
                                                                      X_t=samples,
                                                                      n_projections=100).item()
        print(sliced_wasserstein_distance)
        return {"value": sliced_wasserstein_distance}  # or return dict(value=objective_value)

    def get_objective(self):
        "Returns a dict to pass to the set_objective method of a solver."
        return dict(observation=self.observation,
                    observation_noise=self.observation_noise,
                    forward_operator=self.forward_operator,
                    score_network=self.score_network,
                    alphas_cumprod=self.alphas_cumprod,
                    betas=self.betas,
                    n_samples=self.n_samples)