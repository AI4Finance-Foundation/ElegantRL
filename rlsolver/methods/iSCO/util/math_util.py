import torch

def gumbel(loc):
    uniform_sample = torch.rand(loc.shape,device=loc.device)  # 生成与 loc 相同形状的均匀分布样本
    return loc - torch.log(-torch.log(uniform_sample))  # 计算 Gumbel 分布样本

def log1mexp(x):
    # 计算 log(1 - exp(-|x|))
    x = -torch.abs(x)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))

def noreplacement_sampling_renormalize(ll_idx, dim=-1):
    ll_base = torch.max(ll_idx, dim=dim, keepdim=True).values
    prob_idx = torch.exp(ll_idx - ll_base)
    ll_delta = torch.log(torch.cumsum(prob_idx, dim=dim) - prob_idx) + ll_base
    ll_idx = torch.clamp(ll_idx - log1mexp(ll_delta), max=0.0)
    return ll_idx

def multinomial(log_prob,path_length):
    num_classes = log_prob.shape[-1]
    perturbed_ll = gumbel(log_prob)
    # 将每个点采样的权重升序排列
    sorted_ll, _ = torch.sort(perturbed_ll)
    # 提取排序后的权重的倒数 num_classes - num_samples 列，shape=(batchsize, 1)
    threshold = torch.gather(sorted_ll,1,(num_classes - path_length).unsqueeze(1))
    # threshold = sorted_ll[..., num_classes - path_length].unsqueeze(-1)
    # 在权重最大的 num_samples 个点处为 1，其余为 0
    selected_mask = (perturbed_ll >= threshold.expand_as(perturbed_ll)).int()
    selected = {
        'selected_mask': selected_mask,
        'perturbed_ll': perturbed_ll,
    }
    sorted_idx = torch.argsort(-perturbed_ll, dim=-1)

    # 按照 sorted_idx 的顺序重新排列 log_prob
    sorted_ll = torch.gather(log_prob, dim=-1, index=sorted_idx)

    # 使用 noreplacement_sampling_renormalize 函数对 sorted_ll 进行处理
    idx_ll = noreplacement_sampling_renormalize(sorted_ll)

    # 将 sorted_idx 和 idx_ll 重新调整形状
    flat_idx = sorted_idx.view(-1, num_classes)
    flat_ll = idx_ll.view(-1, num_classes)

    # 初始化 ll_selected 为全零张量
    ll_selected = torch.zeros_like(flat_ll)

    # 根据 flat_idx 的值将 flat_ll 的内容填充到 ll_selected 中
    ll_selected.scatter_(1, flat_idx, flat_ll)

    # 将 ll_selected 重新调整为 log_prob 的形状
    ll_selected = ll_selected.view(log_prob.shape)

    # 应用 selected_mask
    ll_selected = ll_selected * selected_mask

    # selected 是一个字典，其中包含 'selected_mask' 和 'perturbed_ll'


    return selected, ll_selected

def bernoulli_logp(log_prob):
    # 在 PyTorch 中生成形状与 log_prob 相同的均匀分布噪声
    noise = torch.rand(log_prob.shape, device=log_prob.device)
    return torch.log(noise + 1e-24) < log_prob

def mh_step(log_prob, current_sample, new_sample):
    # 使用 log_prob 计算是否使用新样本
    use_new_sample = bernoulli_logp(log_prob)
    # 根据 use_new_sample 决定使用 new_sample 还是 current_sample
    expanded_use_new_sample = use_new_sample.unsqueeze(-1).expand_as(new_sample)
    return (
        torch.where(expanded_use_new_sample, new_sample, current_sample),
        use_new_sample,
    )