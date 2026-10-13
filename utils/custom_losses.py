"""
Defnition for all the training loss functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings


def _select_negative_terms(
    negative_terms,
    hard_negative_mode,
    hard_negative_topk,
    hard_negative_random_k,
    semi_hard_upper,
    device,
):
    """
    Select negative terms according to the configured mining strategy.

    Supported modes:
        - "all": use all negatives (original paper behaviour)
        - "random": randomly select a fixed number of negatives
        - "topk": use the hardest negatives
        - "semi_hard": use only negatives with 0 < violation <= semi_hard_upper
        - "semi_hard_topk": top-k hardest negatives within the semi-hard subset,
          with fallback to regular top-k if the semi-hard subset is empty
    """
    selected_terms = negative_terms

    if hard_negative_mode == "all":
        return selected_terms

    if hard_negative_mode == "random":
        sample_count = min(hard_negative_topk, negative_terms.numel())
        random_indices = torch.randperm(negative_terms.numel(), device=device)[
            :sample_count
        ]
        return negative_terms[random_indices]

    positive_violation_mask = negative_terms > 0

    if hard_negative_mode == "semi_hard":
        semi_hard_mask = positive_violation_mask & (negative_terms <= semi_hard_upper)
        if torch.any(semi_hard_mask):
            return negative_terms[semi_hard_mask]
        return negative_terms[positive_violation_mask] if torch.any(positive_violation_mask) else negative_terms

    candidate_terms = negative_terms
    if hard_negative_mode == "semi_hard_topk":
        semi_hard_mask = positive_violation_mask & (negative_terms <= semi_hard_upper)
        if torch.any(semi_hard_mask):
            candidate_terms = negative_terms[semi_hard_mask]
        elif torch.any(positive_violation_mask):
            candidate_terms = negative_terms[positive_violation_mask]

    if hard_negative_mode in ("topk", "semi_hard_topk"):
        topk = min(hard_negative_topk, candidate_terms.numel())
        _, hard_indices = torch.topk(candidate_terms, k=topk, largest=True)
        selected_indices = hard_indices

        if hard_negative_random_k > 0 and candidate_terms.numel() > topk:
            all_indices = torch.arange(candidate_terms.numel(), device=device)
            remaining_mask = torch.ones_like(all_indices, dtype=torch.bool)
            remaining_mask[hard_indices] = False
            remaining_indices = all_indices[remaining_mask]
            random_k = min(hard_negative_random_k, remaining_indices.numel())
            if random_k > 0:
                random_perm = torch.randperm(remaining_indices.numel(), device=device)
                random_indices = remaining_indices[random_perm[:random_k]]
                selected_indices = torch.cat((hard_indices, random_indices))

        return candidate_terms[selected_indices]

    return selected_terms


def ALM_terms(
    use_ALM,
    buffer_batch_pos,
    buffer_batch_neg,
    lambdas_index_buffer,
    delta,
    mu,
    lambdas,
    train_penalty,
    loss,
    p2_norm,
    device,
    hard_negative_mode="all",
    hard_negative_topk=10,
    hard_negative_random_k=0,
    semi_hard_upper=0.5,
    alm_q_clip=0.0,
    alm_lambda_clip=0.0,
):
    """
    Calculate Augmented Lagrangian Method (ALM) terms for the proposed constraint.

    Parameters:
        use_ALM (int): Flag to indicate whether to use ALM terms (1 for yes, 0 for no).
        buffer_batch_pos (list): List of positive samples.
        buffer_batch_neg (list): List of negative samples.
        lambdas_index_buffer (list): List of lambda index values.
        delta (float): Delta parameter for constraint.
        mu (float): Penalty parameter for ALM.
        lambdas (list): List of lambda values.
        train_penalty (float): Penalty term for training.
        loss (torch.Tensor): Current loss value.
        p2_norm (int): Exponent for the L2 norm in the constraint.
        device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').
        hard_negative_mode (str): Strategy for negative selection. Supported values:
            "all", "random", "topk", "semi_hard", "semi_hard_topk".
        hard_negative_topk (int): Number of hardest negatives selected per positive sample.
        hard_negative_random_k (int): Number of additional random negatives mixed with the
            hard negatives for extra stability.
        semi_hard_upper (float): Upper bound on the violation value for semi-hard negatives.
        alm_q_clip (float): Optional maximum for the aggregate ALM violation.
        alm_lambda_clip (float): Optional absolute cap for each ALM multiplier.

    Returns:
        tuple: A tuple containing:
            - Updated lambdas (list).
            - Train penalty value (float).
            - Updated loss (torch.Tensor).
    """
    relu_function = nn.ReLU()
    for pos_sample, single_lambda_index in zip(buffer_batch_pos, lambdas_index_buffer):
        negative_terms = []
        for neg_sample in buffer_batch_neg:
            negative_terms.append(relu_function(-(pos_sample - neg_sample) + delta).view(1))

        negative_terms = torch.cat(negative_terms)
        selected_terms = _select_negative_terms(
            negative_terms,
            hard_negative_mode,
            hard_negative_topk,
            hard_negative_random_k,
            semi_hard_upper,
            device,
        )

        if not torch.isfinite(selected_terms).all():
            raise FloatingPointError(
                "Non-finite ALM violation in mode={}".format(hard_negative_mode)
            )

        selected_count = max(int(selected_terms.numel()), 1)
        q_pos = torch.sum(selected_terms).view(1, 1)
        if alm_q_clip > 0:
            q_pos = torch.clamp(q_pos, min=0.0, max=alm_q_clip)
        if use_ALM == 1:
            lambda_index = int(single_lambda_index)
            lambda_value = float(lambdas[lambda_index])
            if alm_lambda_clip > 0:
                lambda_value = max(
                    -alm_lambda_clip, min(alm_lambda_clip, lambda_value)
                )
            loss = (
                mu / 2 * torch.pow(q_pos, 2 * p2_norm)
                + lambda_value * q_pos
            ) / (len(buffer_batch_pos) * selected_count)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Non-finite ALM loss")
            # Backward and update the loss with ALM terms
            loss.backward(retain_graph=True)
            # Update Lambda
            updated_lambda = lambda_value + mu * q_pos.detach().item()
            if alm_lambda_clip > 0:
                updated_lambda = max(
                    -alm_lambda_clip,
                    min(alm_lambda_clip, updated_lambda),
                )
            lambdas[lambda_index] = updated_lambda
        # Calculate train penalty, to be analyzed also for training set
        train_penalty += torch.pow(q_pos, 2 * p2_norm).detach().cpu().numpy() / (
            len(buffer_batch_pos) * selected_count
        )
    return lambdas, train_penalty, loss


class SymBCEFocalLoss(nn.Module):
    """
    Symmetric BCE focal loss
    """

    def __init__(self, gamma):
        super(SymBCEFocalLoss, self).__init__()
        self.gamma = gamma

    def asym_focal_loss(self, x, t):  # x = NN output, t = 0/1 label
        FL = torch.zeros_like(t, dtype=float)
        FL[(t).nonzero()] = (
            -torch.pow(1 - torch.sigmoid(x[(t).nonzero()]), self.gamma)
            * torch.log(torch.sigmoid(x[(t).nonzero()]) + 1e-4).double()
        )
        FL[(t == 0).nonzero()] = (
            -torch.pow(torch.sigmoid(x[(t == 0).nonzero()]), self.gamma)
            * torch.log(1 - torch.sigmoid(x[(t == 0).nonzero()]) + 1e-4).double()
        )
        return torch.mean(FL)

    def forward(self, logits, targets):
        FL = self.asym_focal_loss(logits, targets)
        return FL


class SymBCELargeMargin(nn.Module):
    """
    Symmetric BCE margin loss loss
    """

    def __init__(self, m):
        super(SymBCELargeMargin, self).__init__()
        self.m = m

    def sym_large_margin_bce(self, x, t):  # x = NN output, t = 0/1 label
        LM = torch.zeros_like(t, dtype=float)
        LM[(t).nonzero()] = -torch.log(
            torch.sigmoid(x[(t).nonzero()] - self.m) + 1e-4
        ).double()
        LM[(t == 0).nonzero()] = -torch.log(
            1 - torch.sigmoid(x[(t == 0).nonzero()] + self.m) + 1e-4
        ).double()
        # print("check", torch.sigmoid(x-m), 1-torch.sigmoid(x), t)
        return torch.mean(LM)

    def forward(self, logits, targets):
        LM = self.sym_large_margin_bce(logits, targets)
        return LM


class AsymBCEFocalLoss(nn.Module):
    """
    Asymmetric BCE focal loss
    """

    def __init__(self, gamma):
        super(AsymBCEFocalLoss, self).__init__()
        self.gamma = gamma

    def asym_focal_loss(self, x, t):  # x = NN output, t = 0/1 label
        FL = torch.zeros_like(t, dtype=float)
        FL[(t).nonzero()] = -torch.log(torch.sigmoid(x[(t).nonzero()]) + 1e-4).double()
        FL[(t == 0).nonzero()] = (
            -torch.pow(torch.sigmoid(x[(t == 0).nonzero()]), self.gamma)
            * torch.log(1 - torch.sigmoid(x[(t == 0).nonzero()]) + 1e-4).double()
        )
        return torch.mean(FL)

    def forward(self, logits, targets):
        FL = self.asym_focal_loss(logits, targets)
        return FL


class AsymBCELargeMargin(nn.Module):
    """
    Asymmetric BCE margin loss loss
    """

    def __init__(self, m):
        super(AsymBCELargeMargin, self).__init__()
        self.m = m

    def asym_large_margin_bce(self, x, t):  # x = NN output, t = 0/1 label
        LM = torch.zeros_like(t, dtype=float)
        LM[(t).nonzero()] = -torch.log(
            torch.sigmoid(x[(t).nonzero()] - self.m) + 1e-4
        ).double()
        LM[(t == 0).nonzero()] = -torch.log(
            1 - torch.sigmoid(x[(t == 0).nonzero()]) + 1e-4
        ).double()
        return torch.mean(LM)

    def forward(self, logits, targets):
        LM = self.asym_large_margin_bce(logits, targets)
        return LM


class WeightedBCE(nn.Module):
    """
    Weighted BCE
    """

    def __init__(self, c):
        super(WeightedBCE, self).__init__()
        self.c = c

    def weighted_loss(self, x, t):  # x = NN output, t = 0/1 label
        BCE = torch.nn.BCEWithLogitsLoss(reduction="none")
        weight_tensor = torch.ones_like(t, dtype=float)
        weight_tensor[(t).nonzero()] = self.c
        # print(x, weight_tensor, x*weight_tensor)
        weighted_BCE = torch.mean(weight_tensor * BCE(x, t))
        return weighted_BCE

    def forward(self, logits, targets):
        weighted_BCE = self.weighted_loss(logits, targets)
        return weighted_BCE


class ClassBalancedBCE(nn.Module):
    """
    Class balanced BCE
    """

    def __init__(self, beta, n_0, n_1):
        super(ClassBalancedBCE, self).__init__()
        self.beta = beta
        self.n_0 = n_0
        self.n_1 = n_1

    def class_balanced_loss(self, x, t):  # x = NN output, t = 0/1 label
        BCE = torch.nn.BCEWithLogitsLoss(reduction="none")
        weight_tensor = torch.zeros_like(t, dtype=float)
        weight_tensor[(t).nonzero()] = (1 - self.beta) / (1 - self.beta**self.n_1)
        weight_tensor[(t == 0).nonzero()] = (1 - self.beta) / (
            1 - self.beta**self.n_0
        )
        # print(x, weight_tensor, x*weight_tensor)
        cb_BCE = torch.mean(weight_tensor * BCE(x, t))
        return cb_BCE

    def forward(self, logits, targets):
        cb_BCE = self.class_balanced_loss(logits, targets)
        return cb_BCE


class MiniBatchAUC(nn.Module):
    """
    Mini batch AUC
    """

    def __init__(self):
        super(MiniBatchAUC, self).__init__()

    def mini_batch_auc(self, x, t):  # x = NN output, t = 0/1 label
        # x_pos = x[(t).nonzero()]
        # x_neg = x[(t==0).nonzero()]
        if x[(t).nonzero()].size() != 0:
            for i, pos_sample in enumerate(x[(t).nonzero()]):
                # for j, neg_sample in enumerate(x[(t==0).nonzero()]):
                if i == 0:
                    loss = sum(
                        torch.pow(
                            1 - (torch.sigmoid(pos_sample) - torch.sigmoid(x[t == 0])),
                            2,
                        )
                    )
                else:
                    loss += sum(
                        torch.pow(
                            1 - (torch.sigmoid(pos_sample) - torch.sigmoid(x[t == 0])),
                            2,
                        )
                    )
            if len(x[(t).nonzero()]) != 0 and len(x[(t == 0).nonzero()]) != 0:
                loss = loss / (len(x[(t).nonzero()]) * len(x[(t == 0).nonzero()]))
                return torch.sum(loss)

    def forward(self, logits, targets):
        MBAUC = self.mini_batch_auc(logits, targets)
        return MBAUC


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer(
            "m_list", torch.tensor(m_list, dtype=torch.float32)
        )
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1).long(), True)

        index_float = index.to(dtype=x.dtype, device=x.device)
        batch_m = torch.matmul(
            self.m_list.to(device=x.device, dtype=x.dtype)[None, :],
            index_float.transpose(0, 1),
        )
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target.long(), weight=self.weight)


def select_objective_function(
    training_type, gamma, m, c, ratio_pos_train, num_maj, beta, delta, cls_num_list
):
    """
    Function to selct the loss function to be used as objective.
    """
    if training_type == "ASYM_FL":
        loss_function = AsymBCEFocalLoss(gamma)
    elif training_type == "ASYM_LM":
        loss_function = AsymBCELargeMargin(m)
    elif training_type == "SYM_FL":
        loss_function = SymBCEFocalLoss(gamma)
    elif training_type == "SYM_LM":
        loss_function = SymBCELargeMargin(m)
    elif training_type == "WBCE":
        loss_function = WeightedBCE(c * ratio_pos_train)
    elif training_type == "cb_BCE":
        n_0 = num_maj
        n_1 = int(num_maj / ratio_pos_train)
        loss_function = ClassBalancedBCE(beta, n_0, n_1)
    elif training_type == "MBAUC":
        loss_function = MiniBatchAUC()
    elif training_type == "LDAM":
        loss_function = LDAMLoss(
            cls_num_list=cls_num_list, max_m=0.5, s=30, weight=None
        )
    elif training_type == "BCE":
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        warnings.warn("Loss type is not listed")

    return loss_function
