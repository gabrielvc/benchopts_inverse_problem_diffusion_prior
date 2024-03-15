
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from tqdm import tqdm

# Currently, there is no package with DDRM implemented as a method. Thus, here is the extraction of the code necessary to
# run the experiments.
# Taken from https://github.com/bahjat-kawar/ddrm/blob/master/functions/svd_replacement.py
class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                                      1)).view(v.shape[0], M.shape[0])

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


# Taken from https://github.com/bahjat-kawar/ddrm/blob/master/functions/denoising.py
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def efficient_generalized_steps_w_grad(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):

    # setup vectors used in the algorithm
    singulars = H_funcs.singulars()
    Sigma = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], device=x.device)
    Sigma[:singulars.shape[0]] = singulars
    U_t_y = H_funcs.Ut(y_0)
    Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

    # initialize x_T as given in the paper
    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
    large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
    inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
    inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
    inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

    # implement p(x_T | x_0, y) as given in the paper
    # if eigenvalue is too small, we just treat it as zero (only for init)
    init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
    init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[
        large_singulars_index].view(1, -1)
    init_y = init_y.view(*x.size())
    remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
    remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
    init_y = init_y + remaining_s * x
    init_y = init_y / largest_sigmas

    # setup iteration variables
    x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to(x.device)
        if cls_fn == None:
            et = model(xt, t)
        else:
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # variational inference conditioned on y
        sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
        sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
        xt_mod = xt / at.sqrt()[0, 0, 0, 0]
        V_t_x = H_funcs.Vt(xt_mod)
        SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
        V_t_x0 = H_funcs.Vt(x0_t)
        SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

        falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
        cond_before_lite = singulars * sigma_next > sigma_0
        cond_after_lite = singulars * sigma_next < sigma_0
        cond_before = torch.hstack((cond_before_lite, falses))
        cond_after = torch.hstack((cond_after_lite, falses))

        std_nextC = sigma_next * etaC
        sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

        std_nextA = sigma_next * etaA
        sigma_tilde_nextA = torch.sqrt(sigma_next ** 2 - std_nextA ** 2)

        diff_sigma_t_nextB = torch.sqrt(
            sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

        # missing pixels
        Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

        # less noisy than y (after)
        Vt_xt_mod_next[:, cond_after] = \
            V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:,
                                                        cond_after_lite] + std_nextA * torch.randn_like(
                V_t_x0[:, cond_after])

        # noisier than y (before)
        Vt_xt_mod_next[:, cond_before] = \
            (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:,
                                                                      cond_before] + diff_sigma_t_nextB * torch.randn_like(
                U_t_y)[:, cond_before_lite])

        # aggregate all 3 cases and give next prediction
        xt_mod_next = H_funcs.V(Vt_xt_mod_next)
        xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return xs, x0_preds


class Solver(BaseSolver):
    name = 'Python-DDRM'  # proximal gradient, optionally accelerated

    # Any parameter defined here is accessible as an attribute of the solver.
    parameters = {"ddim_params": [(100, .6)]}

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
        dim_y, dim_x = forward_operator.shape
        base_score_module = lambda x, t: -((1 - alphas_cumprod[t]) ** .5) * score_network(x, alphas_cumprod[t])
        model = lambda x, t: base_score_module(x.flatten(1, len(x.shape)-1), t[0].long()).reshape(x.shape[0], 1, 1, dim_x)
        n_steps, eta = self.ddim_params
        H_funcs = GeneralH(H=forward_operator)
        timesteps = torch.linspace(0, 1000, n_steps).long()

        self.sampler = lambda x: efficient_generalized_steps_w_grad(x=x.reshape(n_samples, 1, 1, dim_x),
                                                            b=betas,
                                                            seq=timesteps[:-1].tolist(),
                                                            model=model,
                                                            y_0=observation[None, :],
                                                            H_funcs=H_funcs,
                                                            sigma_0=observation_noise ** .5,
                                                            etaB=1,
                                                            etaA=.85,
                                                            etaC=1,
                                                            classes=None,
                                                            cls_fn=None)[0][-1].flatten(1, 3)
        self.dim_y, self.dim_x = forward_operator.shape
        self.n_samples = n_samples

    def run(self, n_iter):
        initial_samples = torch.randn(size=(self.n_samples, self.dim_x))
        self.samples = self.sampler(initial_samples)

    # Return the solution estimate computed.
    def get_result(self):
        return {'samples': self.samples}