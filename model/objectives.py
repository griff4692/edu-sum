import torch


def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    pad_mask = target.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    target.clamp_min_(0)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    nll_loss.masked_fill_(pad_mask, 0.)
    smooth_loss.masked_fill_(pad_mask, 0.)
    if reduce:
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = pad_mask.numel() - pad_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smooth_loss = smooth_loss.sum() / num_active_elements
    loss = (1. - epsilon) * nll_loss + (epsilon * smooth_loss / lprobs.size(-1))
    return loss, nll_loss


def label_smoothed_unlikelihood(probs, targets, reduce=True):
    probs = probs.view(-1, probs.size(-1))
    one_minus_probs = torch.clamp(1.0 - probs, min=1e-20)
    lprobs = torch.log(one_minus_probs)
    targets = targets.view(-1, 1)
    loss, nll_loss = label_smoothed_nll_loss(lprobs, targets, ignore_index=-100, reduce=reduce)
    return loss, nll_loss
