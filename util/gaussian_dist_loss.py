# Copyright (c) SJTU. All rights reserved.
# modify from https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/gaussian_dist_loss.py
# mirabit

import torch

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5 or _shape[-1] == 4
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = []
    if _shape[-1] == 5:
        r = xywhr[..., 4]
    elif _shape[-1] == 4:
        r = torch.zeros(_shape[0],dtype=xywhr.dtype)
    
    r = r.to(xywhr.device)
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    """Convert oriented bounding box from the Pearson coordinate system to 2-D
    Gaussian distribution.

    Args:
        xy_stddev_pearson (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.square()
    sigma = torch.stack((var[..., 0], covar, covar, var[..., 1]),
                        dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss.

    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance

def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Derivation and simplification:
        Given any positive-definite symmetrical 2*2 matrix Z:
            :math:`Tr(Z^{1/2}) = λ_1^{1/2} + λ_2^{1/2}`
        where :math:`λ_1` and :math:`λ_2` are the eigen values of Z
        Meanwhile we have:
            :math:`Tr(Z) = λ_1 + λ_2`

            :math:`det(Z) = λ_1 * λ_2`
        Combination with following formula:
            :math:`(λ_1^{1/2}+λ_2^{1/2})^2 = λ_1+λ_2+2 *(λ_1 * λ_2)^{1/2}`
        Yield:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
        For gwd loss the frustrating coupling part is:
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σp^{1/2})^{1/2})`
        Assuming :math:`Z = Σ_p^{1/2} * Σ_t * Σ_p^{1/2}` then:
            :math:`Tr(Z) = Tr(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = Tr(Σ_p^{1/2} * Σ_p^{1/2} * Σ_t)
            = Tr(Σ_p * Σ_t)`
            :math:`det(Z) = det(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = det(Σ_p^{1/2}) * det(Σ_t) * det(Σ_p^{1/2})
            = det(Σ_p * Σ_t)`
        and thus we can rewrite the coupling part as:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σ_p^{1/2})^{1/2})
            = (Tr(Σ_p * Σ_t) + 2 * (det(Σ_p * Σ_t))^{1/2})^{1/2}`

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)

def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)