import torch
import torch.nn as nn

@torch.no_grad()
class STPModel(nn.Module):
    def __init__(self, dt=1.0, tau_d=150.0, tau_f=600.0, U=0.2, device='cpu'):
        super().__init__()
        self.dt = dt
        self.device = device

        # 參數
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.U = U

        # 狀態變數
        # self.u = torch.full(pre_neurons, U, device=device)  # facilitation
        # self.x = torch.ones(pre_neurons, device=device)     # depression
        self.u = None
        self.x = None

    def build(self, pre_spikes):
        # lazy init
        self.u = torch.full_like(pre_spikes, self.U, device=self.device)
        self.x = torch.ones_like(pre_spikes, device=self.device)  # [batch, 4096]

    def forward(self, pre_spikes):
        assert self.u is not None, "Call build(pre_spikes) before forward"

        """
        spikes: (num_neurons,) — binary spike tensor
        更新 u(t), x(t) 狀態
        """
        du = ((self.U - self.u) / self.tau_f + self.U * (1 - self.u) * pre_spikes)
        # print(f"du: {du.mean()}, u: {self.u.mean()}, pre_spikes: {pre_spikes.mean()}")
        dx = ((1 - self.x) / self.tau_d - self.u * self.x * pre_spikes)

        inputs_current = self.u * self.x * pre_spikes  # [batch, 4096]
        # print(inputs_current[0][:10])

        self.u = self.u + du * self.dt
        self.x = self.x + dx * self.dt

        return inputs_current  # [batch, 4096]

    def reset(self):
        self.u.fill_(self.U)
        self.x.fill_(1.0)
    
    def get_u(self):
        return self.u
    
    def get_x(self):
        return self.x

@torch.no_grad()
class LTPModel(nn.Module):
    def __init__(self, num_pre, num_post, dt=1.0,
                 eta=0.001, A=1.0,
                 tau_pre=20.0, tau_post_slow=100.0,
                 tau_ht=100.0, tau_hom=10.0 * 60 * 1000,
                 w_p=0.5,w_tilde=0.0,P=20.0,tau_const=20*60*1000.0,
                 w_min=0.0, w_max=5.0, device='cpu'):
        super().__init__()
        self.dt = dt
        self.device = device

        self.num_pre = num_pre
        self.num_post = num_post

        # Triplet STDP parameters
        self.eta = eta
        self.A = A
        self.tau_plus = tau_pre
        self.tau_minus = tau_pre
        self.tau_slow = tau_post_slow
        self.tau_hom = tau_hom
        self.tau_ht = tau_ht

        # traces
        self.z_pre = None
        self.z_post = None
        self.z_post_slow = None
        self.delay_steps = 5  # 要延遲幾個 time step
        self.post_spike_buffer = []
        self.z_ht = None
        self.B = None  # homeostatic variable
        self.C = None  # homeostatic variable
        self.z_post_delay = None
        self.slow_spike = None

        # heterosynaptic parameters
        self.beta = eta * 50.0
        self.w_p = w_p
        self.w_tilde = w_tilde
        self.w_tilde_update_time = 0
        self.P = P
        self.tau_const = tau_const

        # transmitter-induced parameters
        self.delta = eta * 0.02

        # weight limits
        self.w_min = w_min
        self.w_max = w_max
    
    def forward(self, pre_spike, post_spike, weights):
        batch = pre_spike.shape[0]
        # using view and pytorch broadcasting instead of repeat for better performance
        # triplet STDP traces
        positive_STDP = self.A * self.z_pre.view(batch, -1, 1) * self.z_post_slow.view(1, 1, self.num_post) * post_spike
        # STDP
        # positive_STDP = self.A * self.z_pre.view(batch, -1, 1) * post_spike
        negative_STDP = self.B * self.z_post.view(1, 1, self.num_post) * pre_spike.view(batch, -1, 1)
        
        # # triplet STDP weight update
        tripletSTDP = self.eta * (positive_STDP - negative_STDP)

        # using the modulo operator to update w_tilde every 1200 steps
        self.w_tilde_update_time += 1
        if self.w_tilde_update_time % (1200) == 0: # if this w_tilde update time is 1200 or 12000？
            self.update_w_tilde(weights)
            self.w_tilde_update_time = 0
            # print(self.w_tilde.shape)

        heteroSTDP = -self.beta * (weights - self.w_tilde) * (self.z_post_delay ** 3).unsqueeze(0) * post_spike

        # print(f"heteroSTDP: {heteroSTDP.shape}")
        # transmitter-induced weight update
        # using view and pytorch broadcasting instead of repeat for better performance
        transmitterSTDP = self.delta * pre_spike.view(batch, -1, 1)
        # print(f"transmitterSTDP: {transmitterSTDP.shape}")

        # check the value if three types of STDP
        # print(f"tripletSTDP: {tripletSTDP.mean()}, heteroSTDP: {heteroSTDP.mean()}, transmitterSTDP: {transmitterSTDP.mean()}")\
        # check max  (absolute value)
        # print(f"tripletSTDP max: {tripletSTDP.abs().max()}, heteroSTDP max: {heteroSTDP.abs().max()}, transmitterSTDP max: {transmitterSTDP.abs().max()}")

        # weight change
        dw = heteroSTDP + tripletSTDP + transmitterSTDP
        # average over batch size
        dw = dw.sum(dim=0) / pre_spike.shape[0]
        # weights += dw * self.dt
        # weights = torch.clamp(weights, self.w_min, self.w_max)
        weights.data.add_(dw * self.dt)
        weights.data.clamp_(self.w_min, self.w_max)

        self.update_traces(pre_spike, post_spike)
        return weights

    def forward_sparse(self, pre_spike, post_spike, weights, src_idx, dst_idx):
        """
        稀疏版三元 STDP + 異突觸 + transmitter-induced
        只針對 mask 中存在的邊 (src_idx, dst_idx) 做更新，避免建立 [B, Npre, Npost] 巨張量
        """
        device = weights.device
        src_idx = src_idx.to(device)
        dst_idx = dst_idx.to(device)
        Bsz = pre_spike.shape[0]  # batch size
        E = src_idx.numel()       # number of edges

        # --- 取出每條邊對應（邊級）之 pre/post traces & spikes：形狀皆為 [B, E] ---
        z_pre_i       = self.z_pre[:, src_idx]            # [B, E]
        z_post_slow_j = self.z_post_slow[:, dst_idx]      # [B, E]
        z_post_j      = self.z_post[:, dst_idx]           # [B, E]
        z_post_delay_j= self.z_post_delay[:, dst_idx]     # [B, E]
        pre_i         = pre_spike[:, src_idx]             # [B, E]
        post_j        = post_spike[:, dst_idx]            # [B, E]

        # B 是 [B, Npost]，需對齊到邊：取出 [B, E]
        B_j = self.B[:, dst_idx]                          # [B, E]

        # --- Triplet STDP ---
        pos = self.A * z_pre_i * z_post_slow_j * post_j   # [B, E]
        neg = B_j * z_post_j * pre_i                      # [B, E]
        triplet = self.eta * (pos - neg).mean(dim=0)      # [E]

        # --- Transmitter-induced ---
        transmitter = self.delta * pre_i.mean(dim=0)      # [E]

        # --- Heterosynaptic ---
        hetero_mean = (z_post_delay_j ** 3 * post_j).mean(dim=0)  # [E]
        w_ij = weights[src_idx, dst_idx]                           # [E]

        # w_tilde 可能是 scalar 或張量
        if isinstance(self.w_tilde, torch.Tensor) and self.w_tilde.ndim >= 2:
            w_tilde_ij = self.w_tilde[src_idx, dst_idx]            # [E]
        else:
            w_tilde_ij = self.w_tilde                               # scalar

        hetero = -self.beta * (w_ij - w_tilde_ij) * hetero_mean     # [E]

        # --- 合併權重變化，更新到現有邊 ---
        dw_e = triplet + transmitter + hetero                       # [E]
        new_w = (w_ij + dw_e * self.dt).clamp_(self.w_min, self.w_max)
        weights.data.index_put_((src_idx, dst_idx), new_w)

        # 依原設計每 1200 步更新 w_tilde（保持行為一致）
        self.w_tilde_update_time += 1
        if self.w_tilde_update_time % 1200 == 0:
            self.update_w_tilde(weights)
            self.w_tilde_update_time = 0

        # 更新 traces
        self.update_traces(pre_spike, post_spike)
        return weights
    
    def update_traces(self, pre_spike, post_spike):
        # Triplet STDP trace update

        self.post_spike_buffer.append(post_spike.clone())
        if len(self.post_spike_buffer) > self.delay_steps:
            self.post_spike_buffer.pop(0)

        # 使用 delay_steps 前的 spike 來更新 z_post_slow
        delayed_post_spike = self.post_spike_buffer[0]  # time step before epsilon spike
        # z(t - epsilon)
        self.z_post_slow += self.dt * (-self.z_post_slow / self.tau_slow + delayed_post_spike) # epsilon = 1 ms
        self.z_post_delay += self.dt * (-self.z_post / self.tau_minus + delayed_post_spike)
        
        # z(t)
        self.z_pre += self.dt * (-self.z_pre / self.tau_plus + pre_spike)
        self.z_post += self.dt * (-self.z_post / self.tau_minus + post_spike)
        self.z_ht += self.dt * (-self.z_ht / self.tau_ht + post_spike) 
        self.C += self.dt * (-self.C / self.tau_hom + self.z_ht ** 2)  # homeostatic variable update

        # Update B_i(t)
        self.B[self.C <= 1] = self.A * self.C[self.C <= 1]  # Update B where C <= 1
        self.B[self.C > 1] = self.A  # Update B where C > 1
    
    def update_w_tilde(self, weights):
        w_tilde_now = self.w_tilde
        # deferrence equation for w_tilde
        delta = weights - w_tilde_now
        # double well potential
        double_well = self.P * w_tilde_now * (self.w_p / 2 - w_tilde_now) * (self.w_p - w_tilde_now)
        dw_tilde = (delta - double_well) / self.tau_const
        self.w_tilde += dw_tilde * self.dt
    
    def build(self, pre_spike, post_spike):
        # lazy init
        self.z_pre = torch.zeros(pre_spike.shape, device=self.device)
        self.z_post = torch.zeros(post_spike.shape, device=self.device)
        self.z_post_slow = torch.zeros(post_spike.shape, device=self.device)
        self.post_spike_buffer = [torch.zeros(post_spike.shape, device=self.device) for _ in range(self.delay_steps)]
        self.z_ht = torch.zeros(post_spike.shape, device=self.device)
        self.B = torch.ones(post_spike.shape, device=self.device)
        self.C = torch.zeros(post_spike.shape, device=self.device)
        self.z_post_delay = torch.zeros(post_spike.shape, device=self.device)
        self.slow_spike = torch.zeros(post_spike.shape, device=self.device)
        self.delay_post_spike = torch.zeros(post_spike.shape, device=self.device)

@torch.no_grad()
class InhibitorySTDP(nn.Module):
    def __init__(self, dt=1.0,
                 eta_inh=(0.001/50), gamma=4,
                 tau_trace=20.0, tau_H=10 * 1000.0,
                 w_min=0.0, w_max=5.0, device='cpu'):
        super().__init__()
        self.dt = dt
        self.device = device
        self.eta_inh = eta_inh
        self.gamma = (gamma / 1000) * dt
        self.tau_trace = tau_trace
        self.tau_H = tau_H

        self.z_pre = None
        self.z_post = None

        self.H = torch.tensor(0.0, device=device)

        self.w_min = w_min
        self.w_max = w_max

    def forward(self, spikes_pre, spikes_all_brain, weights):
        assert self.z_pre is not None, "Call build(spikes_pre, spikes_post) before forward()"
        z_pre = self.z_pre.unsqueeze(-1)             # [batch, 1024, 1]
        spikes_pre_exp = spikes_pre.unsqueeze(-1)    # [batch, 1024, 1]
        spikes_post_exp = spikes_all_brain.unsqueeze(1)  # [batch, 1, 4096]

        # print(f"spikes_pre: {spikes_pre.shape}, spikes_all_brain: {spikes_all_brain.shape}")
        G = self.H - self.gamma                      # global modulatory factor

        # 更新權重
        z_post_exp = self.z_post.unsqueeze(1)        # [batch, 1, 4096]

        dw = self.eta_inh * G * (
            z_pre * spikes_post_exp + 
            z_post_exp * spikes_pre_exp + 
            spikes_pre_exp
        )

        # average over batch size
        dw = dw.sum(dim=0) / spikes_pre.shape[0]
        
        weights.data.add_(dw * self.dt)
        weights.data.clamp_(self.w_min, self.w_max)

        self.update_traces(spikes_pre, spikes_all_brain)
        return weights

    def forward_sparse(self, spikes_pre, spikes_post_all, weights, src_idx, dst_idx):
        # spikes_pre: [B, Ninhib], spikes_post_all: [B, Nexc]
        B = spikes_pre.shape[0]
        G = self.H - self.gamma

        z_pre_i  = self.z_pre[:, src_idx]         # [B, E]
        z_post_j = self.z_post[:, dst_idx]        # [B, E]
        pre_i    = spikes_pre[:, src_idx]         # [B, E]
        post_j   = spikes_post_all[:, dst_idx]    # [B, E]

        # 原式：dw = eta_inh * G * (z_pre*post + z_post*pre + pre)
        term = (z_pre_i * post_j) + (z_post_j * pre_i) + pre_i
        dw_e = (self.eta_inh * G) * term.mean(dim=0)      # [E]

        w_ij = weights[src_idx, dst_idx]
        weights.data.index_put_((src_idx, dst_idx), (w_ij + dw_e * self.dt).clamp_(self.w_min, self.w_max))

        self.update_traces(spikes_pre, spikes_post_all)
        return weights
    
    def update_traces(self, spikes_pre, spikes_post):
        # 更新 traces
        self.z_pre += self.dt * (-self.z_pre / self.tau_trace + spikes_pre)
        self.z_post += self.dt * (-self.z_post / self.tau_trace + spikes_post)
        self.H += self.dt * (-self.H / self.tau_H + spikes_post.sum())

    def reset(self):
        if self.z_pre is not None:
            self.z_pre.zero_()
        if self.z_post is not None:
            self.z_post.zero_()
        self.H.zero_()
    
    def build(self, spikes_pre, spikes_post):
        self.z_pre = torch.zeros(spikes_pre.shape, device=self.device)
        self.z_post = torch.zeros(spikes_post.shape, device=self.device)

