import torch
import torch.nn as nn
import os
from .synapse_plasticity import STPModel, LTPModel, InhibitorySTDP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

##############################################################################################
#                                         RSNN synapses                                      #
##############################################################################################

# RSNN EE synapse -> LTP + STP
class EE_Synapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons,dt=1.0,
                 # Synapse parameters
                 tau_ampa=5.0, tau_nmda=100.0,
                 alpha_exc=0.2,
                 RSNN_coordinates=None, mode = 'distance',
                 sigma = 1.0,
                shape = 'sphere',
                target_mean=0.05,
                seed=None,
                device='cpu'):
        
        super().__init__()
        self.dt = dt
        self.device = device
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.sigma = sigma

        # 時間常數
        self.tau_ampa = tau_ampa
        self.tau_nmda = tau_nmda
        self.alpha_exc = alpha_exc
        self.alpha_nmda = 1 - self.alpha_exc

        # 初始化狀態值
        # self.g_ampa = torch.zeros(post_neurons, device=device)
        # self.g_nmda = torch.zeros(post_neurons, device=device)
        self.g_ampa = None
        self.g_nmda = None

        # EE synapse -> LTP + STP
        self.stp = STPModel(tau_d=150.0, tau_f=600.0, U=0.2, device=device)
        self.ltp = LTPModel(pre_neurons, post_neurons, device=device)

        # RSNN coordinates
        self.coordinates = RSNN_coordinates
        self.mode = mode
        self.target_mean = target_mean
        self.seed = seed

        # weights
        self.RSNN_EE_weights = torch.rand((pre_neurons, post_neurons), device=device) * 0.2
        self.build_rsnn_rsnn_mask(sigma=self.sigma, target_mean=self.target_mean, shape=shape)
        src, dst = self.RSNN_EE_mask.nonzero(as_tuple=True)
        self.ee_src_idx = src.to(self.device)
        self.ee_dst_idx = dst.to(self.device)
        # self.RSNN_EE_weights = torch.empty((self.pre_neurons, self.post_neurons), device=self.device).uniform_(0.09, 0.11)
        # self.RSNN_EE_weights = 0.1 * torch.ones((self.pre_neurons, self.post_neurons), device=self.device)
        self.RSNN_EE_weights *= self.RSNN_EE_mask  # Apply the mask to the weights
        # visualize_rsnn_connections_animation()
        # self.visualize_rsnn_connections_animation()
    def forward(self, pre_spike):
        """
        spike_exc: (num_pre,) Excitatory 神經元的 spike 輸入
        weights_exc: (num_pre, num_post)
        u, x: STP 的狀態值
        """
        # 更新 AMPA conductance (with STP modulation)
        # print(f"pre_spike: {pre_spike.shape}, self.stp.get_u(): {self.stp.get_u().shape}, self.stp.get_x(): {self.stp.get_x().shape}")
        # (batch_size, 28, 28) -> (batch_size, 1, 784)
        inputs_current = self.stp(pre_spike)  # update STP state, and get effective input
        effective_input = torch.matmul(inputs_current, self.RSNN_EE_weights)
        
        # observe the weights update
        # print(f"RSNN_EE_weights: {self.RSNN_EE_weights.mean()}")
        # self.g_ampa += self.dt * (-self.g_ampa / self.tau_ampa + effective_input)
        # # 更新 NMDA conductance
        # self.g_nmda += self.dt * (-self.g_nmda / self.tau_nmda + self.g_ampa)

        self.g_ampa.add_((effective_input - self.g_ampa / self.tau_ampa) * self.dt)
        self.g_nmda.add_((self.g_ampa - self.g_nmda / self.tau_nmda) * self.dt)

        # 輸出總 excitatory conductance
        g_exc = self.alpha_exc * self.g_ampa + self.alpha_nmda * self.g_nmda

        return g_exc
    
    def update(self, pre_spike, post_spike):
        """
        更新 STP 狀態變數
        """
        # self.ltp(pre_spike, post_spike, self.RSNN_EE_weights)
        # self.RSNN_EE_weights *= self.RSNN_EE_mask  # Apply the mask to the weights 
        
        self.ltp.forward_sparse(pre_spike, post_spike, self.RSNN_EE_weights,
                            self.ee_src_idx, self.ee_dst_idx)
        
    def build_rsnn_rsnn_mask(self, sigma, target_mean=0.05, shape='sphere'):
        """
        建立 RSNN 內部連線遮罩，距離越近連線機率越高
        並保證：
            - 平均連線機率 ≈ target_mean（預設 0.05）
            - 所有連線機率 >= min_prob（預設 0.01）
        """
        tm_str = str(target_mean).replace('.', 'p')
        seed_dir = f"seed_{self.seed}" if self.seed is not None else "default"
        if self.mode == "random":
            mask_path = f"mask/{shape}/random_RSNN/tm_{tm_str}/{seed_dir}/RSNN_EE_mask.pt"
        elif self.mode == "distance":
            mask_path = f"mask/{shape}/sigma_{sigma}/tm_{tm_str}/{seed_dir}/RSNN_EE_mask.pt"
        else:
            mask_path = f"mask/{shape}/fully_connected/{seed_dir}/RSNN_EE_mask.pt"

        if os.path.exists(mask_path):
            self.RSNN_EE_mask = torch.load(mask_path, map_location=self.device)
            print(f"Mask loaded from {mask_path}")
        else:
            if self.mode == "random":
                coords = self.coordinates  # (N, 3)
        
                # 固定連線機率
                prob = torch.full_like(torch.empty(coords.size(0), coords.size(0)), target_mean)

                # 根據機率產生連接遮罩（1: 連線, 0: 不連）
                self.RSNN_EE_mask = torch.bernoulli(prob).to(self.device)
            elif self.mode == "distance":
                coords = self.coordinates  # (N, 3)
                dist = torch.cdist(coords, coords)  # shape: (N, N)

                # 原始 Gaussian 機率
                raw_prob = torch.exp(-dist**2 / (2 * sigma**2))

                # 調整整體平均機率 ≈ target_mean
                # current_mean = prob.mean()
                current_mean = raw_prob.mean()
                scaling_factor = target_mean / current_mean
                prob = raw_prob * scaling_factor

                prob = torch.clamp(prob, min=0.0, max=1.0)  # 確保不超過 1.0

                # 根據機率產生連接遮罩（1: 連線, 0: 不連）
                self.RSNN_EE_mask = torch.bernoulli(prob)
            else:
                # Fully connected mask
                self.RSNN_EE_mask = torch.ones((self.pre_neurons, self.post_neurons), device=self.device)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            # Debug info（可選）
            # print(f"[Mask] RSNN: actual mean = {self.RSNN_EE_mask.float().mean().item():.4f}")
            torch.save(self.RSNN_EE_mask, mask_path)

    
    def reset(self):
        self.g_ampa.zero_()
        self.g_nmda.zero_()
    
    def build(self, pre_spike, post_spike):
        self.stp.build(pre_spike)  # 更新 STP 狀態
        self.ltp.build(pre_spike, post_spike)  # 更新 LTP 狀態
        # inputs_current = (self.stp.u * self.stp.x * pre_spike)
        effective_input = torch.matmul(pre_spike, self.RSNN_EE_weights)
        self.g_ampa = torch.zeros(effective_input.shape, device=self.device)
        self.g_nmda = torch.zeros(effective_input.shape, device=self.device)
        return self.g_ampa
    
    def visualize_rsnn_connections_animation(self, save_path="../film/rsnn_EE_connections.gif", interval=500):
        coords = self.coordinates.cpu().numpy()
        mask = self.RSNN_EE_mask.cpu().numpy()
        num_neurons = coords.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("RSNN Internal Connections Animation")

        # Static background: all RSNN neurons (灰色小點)
        rsnn_background = ax.scatter(*coords.T, c='gray', s=5, alpha=0.1)

        # Dynamic elements
        point_focus = ax.scatter([], [], [], c='red', s=40)  # 被選中的 neuron
        point_connected = ax.scatter([], [], [], c='blue', s=10)  # 它連接到的 neurons
        lines = []

        def init():
            return point_focus, point_connected

        def update(frame):
            nonlocal lines
            for l in lines:
                l.remove()
            lines = []

            idx = frame % num_neurons
            focus_pt = coords[idx]
            connected_idx = mask[idx].nonzero()[0]
            connected_pts = coords[connected_idx]

            point_focus._offsets3d = ([focus_pt[0]], [focus_pt[1]], [focus_pt[2]])
            if connected_pts.shape[0] > 0:
                point_connected._offsets3d = (connected_pts[:, 0], connected_pts[:, 1], connected_pts[:, 2])
            else:
                point_connected._offsets3d = ([], [], [])

            for pt in connected_pts:
                xs, ys, zs = [focus_pt[0], pt[0]], [focus_pt[1], pt[1]], [focus_pt[2], pt[2]]
                line = ax.plot(xs, ys, zs, c='gray', linewidth=0.5, alpha=0.5)[0]
                lines.append(line)

            ax.set_title(f"RSNN EXC 2 EXC neurons {idx} connections")
            return point_focus, point_connected, *lines

        ani = animation.FuncAnimation(
            fig, update, frames=min(16, num_neurons), init_func=init,
            interval=interval, blit=False, repeat=True
        )

        ani.save(save_path, fps=5)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"動畫已儲存：{save_path}")
    
    def show_animation(self, save_path="../film/rsnn_EE_connections.gif", interval=500):
        self.visualize_rsnn_connections_animation(save_path, interval)
        
    
class EI_Synapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons, dt=1.0,
                 tau_ampa=5.0, tau_nmda=100.0,
                 alpha_inh=0.7,
                 mode = 'rsnn',
                 RSNN_Exc_coordinates=None, RSNN_Inh_coordinates=None,
                 sigma=1.0,
                 shape = 'sphere',
                 target_mean=0.05,
                 seed=None,
                 device='cpu'):
        super().__init__()
        self.dt = dt
        self.device = device
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.mode = mode
        self.sigma = sigma
        self.seed = seed

        # 時間常數
        self.tau_ampa = tau_ampa
        self.tau_nmda = tau_nmda
        self.alpha_inh = alpha_inh
        self.alpha_nmda = 1 - self.alpha_inh

        # 初始化狀態值
        self.g_ampa = None
        self.g_nmda = None

        # EI synapse -> STP
        self.stp = STPModel(tau_d=200.0, tau_f=600.0, U=0.2, device=device)

        # RSNN coordinates
        self.RSNN_Exc_coordinates = RSNN_Exc_coordinates
        self.RSNN_Inh_coordinates = RSNN_Inh_coordinates
        self.target_mean = target_mean

        # weights
        self.build_RSNN_exc_to_inh_mask(sigma=self.sigma, target_mean=self.target_mean, shape=shape)
        src, dst = self.RSNN_EI_mask.nonzero(as_tuple=True)
        self.ee_src_idx = src.to(self.device)
        self.ee_dst_idx = dst.to(self.device)
        self.RSNN_EI_weights = torch.rand((self.pre_neurons, self.post_neurons), device=self.device) * 0.4
        # self.RSNN_EI_weights = torch.empty((self.pre_neurons, self.post_neurons), device=self.device).uniform_(0.54, 0.66) # 0.6 ±10%
        # self.RSNN_EI_weights = 0.6 * torch.ones((self.pre_neurons, self.post_neurons), device=self.device)
        self.RSNN_EI_weights *= self.RSNN_EI_mask  # Apply the mask to the weights


        # visualize_rsnn_connections_animation()
        # self.visualize_exc_to_inh_connections_animation()
    def forward(self, pre_spike):
        """
        spike_exc: (num_pre,) Excitatory 神經元的 spike 輸入
        weights_exc: (num_pre, num_post)
        weights_inh: (num_pre, num_post)
        u, x: STP 的狀態值
        """

        # 更新 AMPA conductance (with STP modulation)
        inputs_current = self.stp(pre_spike)  # update STP state, and get effective input
        effective_input = torch.matmul(inputs_current, self.RSNN_EI_weights)

        # self.g_ampa += self.dt * (-self.g_ampa / self.tau_ampa + effective_input)
        # self.g_nmda += self.dt * (-self.g_nmda / self.tau_nmda + self.g_ampa)

        self.g_ampa.add_((effective_input - self.g_ampa / self.tau_ampa) * self.dt)
        self.g_nmda.add_((self.g_ampa - self.g_nmda / self.tau_nmda) * self.dt)

        # 輸出總 excitatory conductance
        g_exc = self.alpha_inh * self.g_ampa + self.alpha_nmda * self.g_nmda

        return g_exc

    def build_RSNN_exc_to_inh_mask(self, sigma, target_mean=0.05, shape='sphere'):
        """
        建立 Exc → Inh 的連線遮罩（Exc 是 pre，Inh 是 post）

        Args:
            sigma (float): Gaussian 控制距離敏感度
            target_mean (float): 希望的平均連接率
            min_prob (float): 所有機率不得低於此值

        Returns:
            mask (torch.Tensor): binary mask，形狀為 (num_exc, num_inh)
        """
        tm_str = str(target_mean).replace('.', 'p')
        seed_dir = f"seed_{self.seed}" if self.seed is not None else "default"
        if self.mode == "random":
            mask_path = f"mask/{shape}/random_RSNN/tm_{tm_str}/{seed_dir}/RSNN_EI_mask.pt"
        elif self.mode == "distance":
            mask_path = f"mask/{shape}/sigma_{sigma}/tm_{tm_str}/{seed_dir}/RSNN_EI_mask.pt"
        else:
            mask_path = f"mask/{shape}/fully_connected/{seed_dir}/RSNN_EI_mask.pt"

        if os.path.exists(mask_path):
            self.RSNN_EI_mask = torch.load(mask_path, map_location=self.device)
            print(f"Mask loaded from {mask_path}")
        else:
            if self.mode == "random":
                # 固定連線機率
                prob = torch.full_like(torch.empty(self.pre_neurons, self.post_neurons), target_mean)
                # 根據機率產生連接遮罩（1: 連線, 0: 不連）
                self.RSNN_EI_mask = torch.bernoulli(prob).to(self.device)
            elif self.mode == "distance":
                pre_coords = self.RSNN_Exc_coordinates  # (num_exc, 3)
                post_coords = self.RSNN_Inh_coordinates  # (num_inh, 3)
                # 計算距離矩陣（每個 Exc neuron 到所有 Inh neurons）
                dist = torch.cdist(pre_coords, post_coords)  # shape: (num_exc, num_inh)

                # 計算連線機率（Gaussian 距離衰減）
                raw_prob = torch.exp(-dist**2 / (2 * sigma**2))

                # 加上最小連線機率限制
                # prob = torch.clamp(raw_prob, min=min_prob)

                # 調整平均值接近 target_mean
                # current_mean = prob.mean()
                current_mean = raw_prob.mean()
                scaling_factor = target_mean / current_mean
                # prob = prob * scaling_factor
                prob = raw_prob * scaling_factor

                # 確保不超過 1.0
                prob = torch.clamp(prob, min = 0.0, max=1.0)

                # 根據機率產生遮罩
                self.RSNN_EI_mask = torch.bernoulli(prob)
            else:
                # Fully connected mask
                self.RSNN_EI_mask = torch.ones((self.pre_neurons, self.post_neurons), device=self.device)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            # Debug info（可選）
            print(f"[Mask] Exc → Inh: actual mean = {self.RSNN_EI_mask.float().mean().item():.4f}")
            torch.save(self.RSNN_EI_mask, mask_path)

    def reset(self):
        self.g_ampa.zero_()
        self.g_nmda.zero_()
    
    def build(self, pre_spike):
        # 更新 AMPA conductance (with STP modulation)
        self.stp.build(pre_spike)  # 更新 STP 狀態
        effective_input = torch.matmul(pre_spike, self.RSNN_EI_weights)
        self.g_ampa = torch.zeros(effective_input.shape, device=self.device)
        self.g_nmda = torch.zeros(effective_input.shape, device=self.device)
        return self.g_nmda

    def visualize_exc_to_inh_connections_animation(self, save_path="../film/rsnn_exc_to_inh_connections.gif", interval=200):
        exc_coords = self.RSNN_Exc_coordinates.cpu().numpy()  # Excitatory neurons
        inh_coords = self.RSNN_Inh_coordinates.cpu().numpy()  # Inhibitory neurons
        mask = self.RSNN_EI_mask.cpu().numpy()                # Exc → Inh mask
        num_exc = exc_coords.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Exc → Inh Connections Animation")

        # Static background (灰色半透明小點)
        ax.scatter(*exc_coords.T, c='lightblue', s=5, alpha=0.1, label='Exc neurons')
        ax.scatter(*inh_coords.T, c='lightcoral', s=5, alpha=0.1, label='Inh neurons')

        # Dynamic elements
        point_exc = ax.scatter([], [], [], c='blue', s=40)  # Exc neuron (focus)
        point_inh = ax.scatter([], [], [], c='red', s=10)   # Connected Inh neurons
        lines = []

        def init():
            return point_exc, point_inh

        def update(frame):
            nonlocal lines
            for l in lines:
                l.remove()
            lines = []

            idx = frame % num_exc
            exc_pt = exc_coords[idx]
            connected_idx = mask[idx].nonzero()[0]
            connected_pts = inh_coords[connected_idx]

            point_exc._offsets3d = ([exc_pt[0]], [exc_pt[1]], [exc_pt[2]])
            if connected_pts.shape[0] > 0:
                point_inh._offsets3d = (connected_pts[:, 0], connected_pts[:, 1], connected_pts[:, 2])
            else:
                point_inh._offsets3d = ([], [], [])

            for pt in connected_pts:
                xs, ys, zs = [exc_pt[0], pt[0]], [exc_pt[1], pt[1]], [exc_pt[2], pt[2]]
                line = ax.plot(xs, ys, zs, c='gray', linewidth=0.5, alpha=0.5)[0]
                lines.append(line)

            ax.set_title(f"Exc neuron {idx} connections to Inh neurons")
            return point_exc, point_inh, *lines

        ani = animation.FuncAnimation(
            fig, update, frames=min(16, num_exc), init_func=init,
            interval=interval, blit=False, repeat=True
        )

        ani.save(save_path, fps=5)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"動畫已儲存：{save_path}")

    def show_animation(self, save_path="../film/rsnn_exc_to_inh_connections.gif", interval=200):
        self.visualize_exc_to_inh_connections_animation(save_path, interval)

# IE synapse
class IE_Synapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons, dt=1.0,
                 tau_gaba=10.0, tau_adapt=100.0,
                 delta_a=0.1,
                 mode = 'rsnn',
                 RSNN_Exc_coordinates=None, RSNN_Inh_coordinates=None,
                 sigma=1.0,
                 shape = 'sphere',
                 target_mean=0.05,
                 seed=None,
                 device='cpu'):

        super().__init__()
        self.dt = dt
        self.device = device
        self.pre_neurons = pre_neurons
        self.seed = seed
        self.post_neurons = post_neurons
        self.mode = mode
        self.sigma = sigma

        # 時間常數
        self.tau_gaba = tau_gaba
        self.tau_adapt = tau_adapt
        self.delta_a = delta_a  # adaptation strength

        # 初始化狀態值
        # self.g_gaba = torch.zeros(pre_neurons, device=device)
        # self.g_adapt = torch.zeros(pre_neurons, device=device)
        self.g_gaba = None
        self.g_adapt = None

        # RSNN coordinates
        self.RSNN_Exc_coordinates = RSNN_Exc_coordinates
        self.RSNN_Inh_coordinates = RSNN_Inh_coordinates
        self.target_mean = target_mean

        # RSNN inhibitory neurons to RSNN excitatory neurons
        self.build_RSNN_inh_to_exc_mask(sigma=self.sigma, target_mean=self.target_mean, shape=shape)
        src, dst = self.IE_weights_mask.nonzero(as_tuple=True)
        self.ie_src_idx = src.to(self.device)
        self.ie_dst_idx = dst.to(self.device)
        self.IE_weights = torch.rand((self.pre_neurons, self.post_neurons), device=device) * 0.3
        # self.IE_weights = torch.empty((self.pre_neurons, self.post_neurons), device=self.device).uniform_(0.18, 0.22) # 0.2 ±10%
        # self.IE_weights = 0.2 * torch.ones((self.pre_neurons, self.post_neurons), device=self.device)
        self.IE_weights *= self.IE_weights_mask  # Apply the mask to the weights


        # visualize_rsnn_connections_animation()
        # self.visualize_inh_to_exc_connections_animation()
        # inhibitory STDP
        self.inhibitory_stdp = InhibitorySTDP(device=device)

    def forward(self, pre_spike):
        """
        spike_inh: (num_pre,) Inhibitory 神經元的 spike 輸入
        weights_inh: (num_pre, num_post)
        """

        # 更新 GABA conductance
        input_gaba = torch.matmul(pre_spike, self.IE_weights)  # inhibitory inputs
        self.g_gaba += self.dt * (-self.g_gaba / self.tau_gaba + input_gaba)

        return self.g_gaba
    
    ## call when post neuron spikes in forward pass
    def update_g_adapt(self, post_spike):
        self.g_adapt += self.dt * (-self.g_adapt / self.tau_adapt + self.delta_a * post_spike)
        return self.g_adapt

    def build_RSNN_inh_to_exc_mask(self, sigma, target_mean=0.05, shape='sphere'):
        """
        建立 Inh → Exc 的連線遮罩（Inh 是 pre，Exc 是 post）

        Args:
            sigma (float): Gaussian 控制距離敏感度
            target_mean (float): 希望的平均連接率
            min_prob (float): 所有機率不得低於此值

        Returns:
            mask (torch.Tensor): binary mask，形狀為 (num_exc, num_inh)
        """
        tm_str = str(target_mean).replace('.', 'p')
        seed_dir = f"seed_{self.seed}" if self.seed is not None else "default"
        if self.mode == "random":
            mask_path = f"mask/{shape}/random_RSNN/tm_{tm_str}/{seed_dir}/RSNN_IE_mask.pt"
        elif self.mode == "distance":
            mask_path = f"mask/{shape}/sigma_{sigma}/tm_{tm_str}/{seed_dir}/RSNN_IE_mask.pt"
        else:
            mask_path = f"mask/{shape}/fully_connected/{seed_dir}/RSNN_IE_mask.pt"

        if os.path.exists(mask_path):
            self.IE_weights_mask = torch.load(mask_path, map_location=self.device)
            print(f"Mask loaded from {mask_path}")
        else:
            if self.mode == "random":
                # 固定連線機率
                prob = torch.full_like(torch.empty(self.pre_neurons, self.post_neurons), target_mean)
                # 根據機率產生連接遮罩（1: 連線, 0: 不連）
                self.IE_weights_mask = torch.bernoulli(prob).to(self.device)
            elif self.mode == "distance":
                pre_coords = self.RSNN_Inh_coordinates  # (num_inh, 3)
                post_coords = self.RSNN_Exc_coordinates  # (num_exc, 3)

                # 計算距離矩陣（每個 Inh neuron 到所有 Exc neurons）
                dist = torch.cdist(pre_coords, post_coords)  # shape: (num_inh, num_exc)

                # 計算連線機率（Gaussian 距離衰減）
                raw_prob = torch.exp(-dist**2 / (2 * sigma**2))

                # 加上最小連線機率限制
                # prob = torch.clamp(raw_prob, min=min_prob)

                # 調整平均值接近 target_mean
                # current_mean = prob.mean()
                current_mean = raw_prob.mean()
                scaling_factor = target_mean / current_mean
                # prob = prob * scaling_factor
                prob = raw_prob * scaling_factor

                # 確保不超過 1.0
                prob = torch.clamp(prob, min=0.0, max=1.0)

                # 根據機率產生遮罩
                self.IE_weights_mask = torch.bernoulli(prob)
            else:
                # Fully connected mask
                self.IE_weights_mask = torch.ones((self.pre_neurons, self.post_neurons), device=self.device)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            # Debug info（可選）
            # print(f"[Mask] Inh → Exc: actual mean = {mask.float().mean().item():.4f}")
            torch.save(self.IE_weights_mask, mask_path)

    def update(self, pre_spike, post_spike):
        """
        更新 inhibitory_stdp 狀態變數
        """
        # self.inhibitory_stdp(pre_spike, post_spike, self.IE_weights)
        # self.IE_weights *= self.IE_weights_mask  # Apply the mask to the weights
        self.inhibitory_stdp.forward_sparse(pre_spike, post_spike, self.IE_weights,
                                    self.ie_src_idx, self.ie_dst_idx)

    def reset(self):
        self.g_gaba.zero_()
        self.g_adapt.zero_()

    def build(self, pre_spike, post_spike):
        self.inhibitory_stdp.build(pre_spike, post_spike)
        # 更新 GABA conductance
        input_gaba = torch.matmul(pre_spike, self.IE_weights)  # inhibitory inputs
        self.g_gaba = torch.zeros(input_gaba.shape, device=self.device)
        self.g_adapt = torch.zeros(input_gaba.shape, device=self.device)
        
        return self.g_gaba, self.g_adapt
    
    def visualize_inh_to_exc_connections_animation(self, save_path="../film/rsnn_inh_to_exc_connections.gif", interval=200):
        inh_coords = self.RSNN_Inh_coordinates.cpu().numpy()  # Inhibitory neurons
        exc_coords = self.RSNN_Exc_coordinates.cpu().numpy()  # Excitatory neurons
        mask = self.IE_weights_mask.cpu().numpy()             # Inh → Exc mask
        num_inh = inh_coords.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Inh → Exc Connections Animation")

        # Static background (半透明小點)
        ax.scatter(*inh_coords.T, c='lightcoral', s=5, alpha=0.1, label='Inh neurons')
        ax.scatter(*exc_coords.T, c='skyblue', s=5, alpha=0.1, label='Exc neurons')

        # Dynamic elements
        point_inh = ax.scatter([], [], [], c='red', s=40)   # Inhibitory neuron (focus)
        point_exc = ax.scatter([], [], [], c='blue', s=10)  # Connected Excitatory neurons
        lines = []

        def init():
            return point_inh, point_exc

        def update(frame):
            nonlocal lines
            for l in lines:
                l.remove()
            lines = []

            idx = frame % num_inh
            inh_pt = inh_coords[idx]
            connected_idx = mask[idx].nonzero()[0]
            connected_pts = exc_coords[connected_idx]

            point_inh._offsets3d = ([inh_pt[0]], [inh_pt[1]], [inh_pt[2]])
            if connected_pts.shape[0] > 0:
                point_exc._offsets3d = (connected_pts[:, 0], connected_pts[:, 1], connected_pts[:, 2])
            else:
                point_exc._offsets3d = ([], [], [])

            for pt in connected_pts:
                xs, ys, zs = [inh_pt[0], pt[0]], [inh_pt[1], pt[1]], [inh_pt[2], pt[2]]
                line = ax.plot(xs, ys, zs, c='gray', linewidth=0.5, alpha=0.5)[0]
                lines.append(line)

            ax.set_title(f"Inh neuron {idx} connections to Exc neurons")
            return point_inh, point_exc, *lines

        ani = animation.FuncAnimation(
            fig, update, frames=min(16, num_inh), init_func=init,
            interval=interval, blit=False, repeat=True
        )

        ani.save(save_path, fps=5)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"動畫已儲存：{save_path}")
    
    def show_animation(self, save_path="../film/rsnn_inh_to_exc_connections.gif", interval=200):
        self.visualize_inh_to_exc_connections_animation(save_path, interval)

# Inhibitory 2 Inhibitory synapse -> Static
# RSNN only
class II_Synapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons, dt=1.0,
                 tau_gaba=10.0, tau_adapt=100.0,
                 delta_a=0.1,
                 RSNN_coordinates=None, mode='rsnn',
                 sigma=1.0,
                 shape = 'sphere',
                 target_mean=0.05,
                 seed=None,
                 device='cpu'):

        super().__init__()
        self.dt = dt
        self.device = device
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.seed = seed
        self.mode = mode
        self.sigma = sigma

        # 時間常數
        self.tau_gaba = tau_gaba
        self.tau_adapt = tau_adapt
        self.delta_a = delta_a  # adaptation strength

        # 初始化狀態值
        # self.g_gaba = torch.zeros(pre_neurons, device=device)
        # self.g_adapt = torch.zeros(pre_neurons, device=device)
        self.g_gaba = None
        self.g_adapt = None

        self.RSNN_coordinates = RSNN_coordinates
        self.target_mean = target_mean

        # RSNN inhibitory neurons to RSNN inhibitory neurons
        # self.build_weights_mask(sigma = self.sigma, shape=shape)
        # self.II_weights = torch.rand((self.pre_neurons, self.post_neurons), device=device) * 0.4
        # # self.II_weights = torch.empty((self.pre_neurons, self.post_neurons), device=self.device).uniform_(0.18, 0.22) # 0.2 ±10%
        # # self.II_weights = 0.2 * torch.ones((self.pre_neurons, self.post_neurons), device=self.device)

        self.build_weights_mask(sigma=self.sigma, target_mean=self.target_mean, shape=shape)
        src, dst = self.II_weights_mask.nonzero(as_tuple=True)
        self.ee_src_idx = src.to(self.device)
        self.ee_dst_idx = dst.to(self.device)
        self.II_weights = torch.rand((self.pre_neurons, self.post_neurons), device=device) * 0.4
        self.II_weights *= self.II_weights_mask  # Apply the mask to the weights

        # visualize_rsnn_connections_animation()
        # self.visualize_inh_to_inh_connections_animation()

    def forward(self, pre_spike):
        """
        spike_inh: (num_pre,) Inhibitory 神經元的 spike 輸入
        weights_inh: (num_pre, num_post)
        """

        # 更新 GABA conductance
        input_gaba = torch.matmul(pre_spike, self.II_weights)  # inhibitory inputs
        self.g_gaba += self.dt * (-self.g_gaba / self.tau_gaba + input_gaba)

        return self.g_gaba
    
    def update_g_adapt(self, post_spike):
        self.g_adapt += self.dt * (-self.g_adapt / self.tau_adapt + self.delta_a * post_spike)
        return self.g_adapt

    def build_weights_mask(self, sigma, target_mean=0.05, shape='sphere'):
        tm_str = str(target_mean).replace('.', 'p')
        seed_dir = f"seed_{self.seed}" if self.seed is not None else "default"
        if self.mode == "random":
            mask_path = f"mask/{shape}/random_RSNN/tm_{tm_str}/{seed_dir}/RSNN_II_mask.pt"
        elif self.mode == "distance":
            mask_path = f"mask/{shape}/sigma_{sigma}/tm_{tm_str}/{seed_dir}/RSNN_II_mask.pt"
        else: # Fully connected
            mask_path = f"mask/{shape}/fully_connected/{seed_dir}/RSNN_II_mask.pt"

        if os.path.exists(mask_path):
            self.II_weights_mask = torch.load(mask_path, map_location=self.device)
            print(f"Mask loaded from {mask_path}")
        else:
            if self.mode == "random":
                # 固定連線機率
                prob = torch.full_like(torch.empty(self.pre_neurons, self.post_neurons), target_mean)
                # 根據機率產生連接遮罩（1: 連線, 0: 不連）
                self.II_weights_mask = torch.bernoulli(prob).to(self.device)
            elif self.mode == "distance":
                dist = torch.cdist(self.RSNN_coordinates, self.RSNN_coordinates)  # shape: (N, N)

                # 原始 Gaussian 機率
                raw_prob = torch.exp(-dist**2 / (2 * sigma**2))

                # 加上下限限制：避免小於 min_prob
                # prob = torch.clamp(raw_prob, min=min_prob)

                # 調整整體平均機率 ≈ target_mean
                # current_mean = prob.mean()
                current_mean = raw_prob.mean()
                scaling_factor = target_mean / current_mean
                # prob = prob * scaling_factor
                prob = raw_prob * scaling_factor

                # 再次 clamp，確保所有值 ∈ [min_prob, 1.0]
                prob = torch.clamp(prob, min = 0.0, max=1.0)

                # 根據機率產生連接遮罩（1: 連線, 0: 不連）
                self.II_weights_mask = torch.bernoulli(prob)
            
            else:  # Fully connected
                # 全連接情況下，直接設置為全 1
                self.II_weights_mask = torch.ones((self.pre_neurons, self.post_neurons), device=self.device)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            torch.save(self.II_weights_mask, mask_path)

    def reset(self):
        self.g_gaba.zero_()
        self.g_adapt.zero_()

    def build(self, pre_spike):
        input_gaba = torch.matmul(pre_spike, self.II_weights)  # inhibitory inputs
        self.g_gaba = torch.zeros(input_gaba.shape, device=self.device)
        self.g_adapt = torch.zeros(input_gaba.shape, device=self.device)
        return self.g_gaba, self.g_adapt

    def visualize_inh_to_inh_connections_animation(self, save_path="../film/rsnn_inh_to_inh_connections.gif", interval=200):
        inh_coords = self.RSNN_coordinates.cpu().numpy()  # Inhibitory neurons (both pre and post)
        mask = self.II_weights_mask.cpu().numpy()         # Inh → Inh mask
        num_inh = inh_coords.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Inh → Inh Connections Animation")

        # Static background
        ax.scatter(*inh_coords.T, c='gray', s=5, alpha=0.1, label='Inh neurons')

        # Dynamic elements
        point_inh = ax.scatter([], [], [], c='red', s=40)    # Current Inh neuron
        point_connected = ax.scatter([], [], [], c='blue', s=10)  # Connected Inh neurons
        lines = []

        def init():
            return point_inh, point_connected

        def update(frame):
            nonlocal lines
            for l in lines:
                l.remove()
            lines = []

            idx = frame % num_inh
            inh_pt = inh_coords[idx]
            connected_idx = mask[idx].nonzero()[0]
            connected_pts = inh_coords[connected_idx]

            point_inh._offsets3d = ([inh_pt[0]], [inh_pt[1]], [inh_pt[2]])
            if connected_pts.shape[0] > 0:
                point_connected._offsets3d = (connected_pts[:, 0], connected_pts[:, 1], connected_pts[:, 2])
            else:
                point_connected._offsets3d = ([], [], [])

            for pt in connected_pts:
                xs, ys, zs = [inh_pt[0], pt[0]], [inh_pt[1], pt[1]], [inh_pt[2], pt[2]]
                line = ax.plot(xs, ys, zs, c='gray', linewidth=0.5, alpha=0.5)[0]
                lines.append(line)

            ax.set_title(f"Inh neuron {idx} connections to Inh neurons")
            return point_inh, point_connected, *lines

        ani = animation.FuncAnimation(
            fig, update, frames=min(16, num_inh), init_func=init,
            interval=interval, blit=False, repeat=True
        )

        ani.save(save_path, fps=5)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"動畫已儲存：{save_path}")
    
    def show_animation(self, save_path="../film/rsnn_inh_to_inh_connections.gif", interval=200):
        self.visualize_inh_to_inh_connections_animation(save_path, interval)


#############################################################################################
#                                   Inputs Synapse                                          #
#############################################################################################

# inputs 2 RSNN synapse
class inputs_2_RSNN_EESynapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons,dt=1.0,
                 # Synapse parameters
                 tau_ampa=5.0, tau_nmda=100.0,
                 alpha_exc=0.2,
                 Inputs_coordinates = None, RSNN_coordinates=None,
                 shape = 'sphere',
                 inputs_shape=None,
                 input_mode='spatial',
                 seed=None,
                device='cpu'):
        
        super().__init__()
        self.dt = dt
        self.device = torch.device(device)
        # pre_neurons count = H x W
        # pre_neutons = [batch, 1, H, W]
        self.pre_neurons = pre_neurons[1] * pre_neurons[2] * pre_neurons[3]
        self.post_neurons = post_neurons
        self.inputs_shape = inputs_shape
        self.seed = seed

        # 時間常數
        self.tau_ampa = tau_ampa
        self.tau_nmda = tau_nmda
        self.alpha_exc = alpha_exc

        # 初始化狀態值
        # self.g_ampa = torch.zeros(post_neurons, device=device)
        # self.g_nmda = torch.zeros(post_neurons, device=device)
        self.g_ampa = None
        self.g_nmda = None

        # EE synapse -> LTP + STP
        self.stp = STPModel(tau_d=150.0, tau_f=600.0, U=0.2, device=device)
        self.ltp = LTPModel(pre_neurons, post_neurons, device=device)

        # inputs coordinates
        self.inputs_coordinates = Inputs_coordinates
        # RSNN coordinates
        self.RSNN_coordinates = RSNN_coordinates
        if input_mode != 'spatial':
            raise ValueError(f"input_mode must be 'spatial', got {input_mode!r}")
        self.input_mode = input_mode  # 'spatial'

        # weights
        self.build_input_to_rsnn_mask(shape=shape)
        # self.inputs_EE_weights = torch.empty((self.pre_neurons, self.post_neurons), device=self.device).uniform_(0.45, 0.55) # 0.5 ±10%
        self.inputs_EE_weights = torch.rand((self.pre_neurons, self.post_neurons), device=device)
        # self.inputs_EE_weights = 0.5 * torch.ones((self.pre_neurons, self.post_neurons), device=self.device)
        self.inputs_EE_weights *= self.inputs_EE_mask  # Apply the mask to the weights

        # self.visualize_single_rf()
        # self.visualize_rf_animation()
        # self.animate_input_to_rsnn()
    
    def forward(self, pre_spike):
        """
        spike_exc: (num_pre,) Excitatory 神經元的 spike 輸入
        weights_exc: (num_pre, num_post)
        u, x: STP 的狀態值
        """
        # 更新 AMPA conductance (with STP modulation)
        # (batch_size, 28, 28) -> (batch_size, 1, 784)

        inputs_current = self.stp(pre_spike).view(pre_spike.shape[0], -1)  # (batch_size, 784)
        effective_input = torch.matmul(inputs_current, self.inputs_EE_weights)
        # print(effective_input.shape) # (batch_size, post_neurons)

        self.g_ampa.add_((effective_input - self.g_ampa / self.tau_ampa) * self.dt)
        self.g_nmda.add_((self.g_ampa - self.g_nmda / self.tau_nmda) * self.dt)

        # 輸出總 excitatory conductance
        g_exc = self.alpha_exc * self.g_ampa + (1 - self.alpha_exc) * self.g_nmda

        return g_exc
    
    def update(self, pre_spike, post_spike):
        """
        更新 STP 狀態變數
        """
        self.ltp(pre_spike, post_spike, self.inputs_EE_weights)
        self.inputs_EE_weights *= self.inputs_EE_mask  # Apply the mask to the weights
            
    def build_input_to_rsnn_mask(self, top_k=250, shape='sphere'):
        """
        input layer 是平面, RSNN 是球型。對應方式是將 input 的位置映射到球體側面，
        然後根據空間位置距離建立連結遮罩。

        input_mode='spatial': 每個 input neuron 找最近的 RSNN neuron 當中心 (原始設計)
        """
        mode_tag = self.input_mode  # 'spatial'
        seed_dir = f"seed_{self.seed}" if self.seed is not None else "default"
        mask_path = f"mask/{self.inputs_shape}/{shape}/{top_k}/{mode_tag}/{seed_dir}/inputs_2_RSNN_EE_mask.pt"
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        if os.path.exists(mask_path):
            self.inputs_EE_mask = torch.load(mask_path, map_location=self.device)
            print(f"load inputs2EE Mask ({mode_tag}):", {mask_path})
            return

        input_coords = self.inputs_coordinates  # [N_in, 3]
        rsnn_coords = self.RSNN_coordinates     # [N_exc, 3]
        N_in = input_coords.shape[0]
        N_exc = rsnn_coords.shape[0]
        device = self.device

        mask = torch.zeros((N_in, N_exc), device=device)

        rsnn_x = rsnn_coords[:, 0]
        rsnn_y = rsnn_coords[:, 1]
        rsnn_z = rsnn_coords[:, 2]

        for i in range(N_in):
            in_x, in_y, in_z = input_coords[i]

            # Spatial center: find the nearest RSNN neuron at z≈0
            xy_diff = torch.norm(rsnn_coords[:, :2] - input_coords[i, :2], dim=1)
            z_penalty = torch.abs(rsnn_z)
            composite_score = xy_diff + 0.5 * z_penalty
            center_idx = torch.argmin(composite_score).item()

            center = rsnn_coords[center_idx]

            # 以Z軸較長的橢圓找最近的 top_k 顆 neuron
            diff = rsnn_coords - center
            scale = torch.tensor([1.0, 1.0, 2], device=device)
            scaled_diff = diff / scale
            dist_to_center = torch.norm(scaled_diff, dim=1)

            topk_idx = torch.topk(dist_to_center, k=top_k, largest=False).indices

            # Spatial mode: apply directional filtering (original logic)
            left_right = (in_x - rsnn_x[topk_idx])
            up_down = (in_y - rsnn_y[topk_idx])

            valid_idx = topk_idx[
                ((in_x <= 0) & (left_right > 0)) |
                ((in_x > 0) & (left_right < 0)) |
                ((in_y <= 0) & (up_down > 0)) |
                ((in_y > 0) & (up_down < 0))
            ]

            if len(valid_idx) > top_k:
                valid_idx = valid_idx[:top_k]
            elif len(valid_idx) < top_k:
                extra = [j for j in topk_idx if j not in valid_idx][:top_k - len(valid_idx)]
                valid_idx = torch.cat([valid_idx, torch.tensor(extra, device=device)])

            mask[i, valid_idx] = 1.0

        self.inputs_EE_mask = mask
        print(self.inputs_EE_mask.shape)
        print(f"[Mask] Input to RSNN ({mode_tag}): actual mean = {mask.float().mean().item():.4f}")
        torch.save(mask, mask_path)
    
    def visualize_rf_animation(self, save_path="../film/receptive_field.gif", interval=200):
        input_coords = self.inputs_coordinates.cpu().numpy()
        rsnn_coords = self.RSNN_coordinates.cpu().numpy()
        mask = self.inputs_EE_mask.cpu().numpy()
        num_inputs = input_coords.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 2.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Receptive Field Animation")

        # Static (faint) background: all neurons
        ax.scatter(*input_coords.T, c='skyblue', s=3, alpha=0.1)
        ax.scatter(*rsnn_coords.T, c='lightcoral', s=3, alpha=0.1)

        # Dynamic elements
        point_input = ax.scatter([], [], [], c='blue', s=40)
        point_rsnn = ax.scatter([], [], [], c='red', s=10)
        lines = []

        def init():
            return point_input, point_rsnn

        def update(frame):
            nonlocal lines
            for l in lines:
                l.remove()
            lines = []

            idx = frame % num_inputs
            input_pt = input_coords[idx]
            connected_idx = mask[idx].nonzero()[0]
            connected_pts = rsnn_coords[connected_idx]

            point_input._offsets3d = ([input_pt[0]], [input_pt[1]], [input_pt[2]])
            if connected_pts.shape[0] > 0:
                point_rsnn._offsets3d = (connected_pts[:, 0], connected_pts[:, 1], connected_pts[:, 2])
            else:
                point_rsnn._offsets3d = ([], [], [])

            for pt in connected_pts:
                xs, ys, zs = [input_pt[0], pt[0]], [input_pt[1], pt[1]], [input_pt[2], pt[2]]
                line = ax.plot(xs, ys, zs, c='gray', linewidth=0.5, alpha=0.5)[0]
                lines.append(line)

            ax.set_title(f"Input neuron {idx} receptive field")
            return point_input, point_rsnn, *lines

        ani = animation.FuncAnimation(fig, update, frames=min(300, num_inputs), init_func=init,
                                    interval=interval, blit=False, repeat=True)

        # ani = animation.FuncAnimation(fig, update, frames=min(28, num_inputs), init_func=init,
        #                         interval=interval, blit=False, repeat=True)

        ani.save(save_path, fps=5)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"動畫已儲存：{save_path}")

    def reset(self):
        self.g_ampa.zero_()
        self.g_nmda.zero_()
    
    def build(self, pre_spike, post_spike):
        self.stp.build(pre_spike)
        self.ltp.build(pre_spike, post_spike)
        inputs_current = (pre_spike).view(pre_spike.shape[0], -1)  # (batch_size, 784)
        effective_input = torch.matmul(inputs_current, self.inputs_EE_weights)
        self.g_ampa = torch.zeros(effective_input.shape, device=self.device)
        self.g_nmda = torch.zeros(effective_input.shape, device=self.device)
        return self.g_nmda
    
    def show_animation(self, save_path="../film/inputs_2_RSNN.gif", interval=200):
        self.visualize_rf_animation(save_path, interval)


# inputs EI synapse
class inputs_EI_Synapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons, dt=1.0, 
                 tau_ampa=5.0, tau_nmda=100.0,
                 alpha_inh=0.3,
                 inputs_shape=(1, 1, 28, 28),
                 device='cpu'):
                 
        super().__init__()
        self.dt = dt
        self.device = device
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.inputs_shape = inputs_shape  # (batch_size, channels, height, width)

        # 時間常數
        self.tau_ampa = tau_ampa
        self.tau_nmda = tau_nmda
        self.alpha_inh = alpha_inh

        # 初始化狀態值
        # self.g_ampa = torch.zeros(post_neurons, device=device)
        # self.g_nmda = torch.zeros(post_neurons, device=device)
        self.g_ampa = None
        self.g_nmda = None

        # EI synapse -> STP
        self.stp = STPModel(tau_d=200.0, tau_f=600.0, U=0.2, device=device)

        self.exc_to_inh = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.exc_to_inh.weight.data.fill_(0.6)
        # self.exc_to_inh.weight.data.uniform_(0.58, 0.62)

    def _resolve_inputs_shape(self, pre_spike):
        """
        Resolve the input image shape for the current batch.

        Historically this synapse stored a fixed `inputs_shape=(1, 1, 28, 28)`
        and reshaped all inputs to that exact shape, which silently assumes
        batch_size=1. For batched evaluation we keep the spatial dimensions but
        infer the current batch dimension from `pre_spike`.
        """
        if len(self.inputs_shape) != 4:
            raise ValueError(f"inputs_shape must be 4D, got {self.inputs_shape}")
        return (pre_spike.shape[0], *self.inputs_shape[1:])

    def forward(self, pre_spike):
        """
        spike_exc: (num_pre,) Excitatory 神經元的 spike 輸入
        weights_exc: (num_pre, num_post)
        weights_inh: (num_pre, num_post)
        u, x: STP 的狀態值
        """

        # 更新 AMPA conductance (with STP modulation)
        # effective_input = torch.matmul((self.stp.get_u() * self.stp.get_x() * pre_spike), self.inputs_EI_weights)
        # --- 插入 STP 計算 ---
        effective_spike = self.stp(pre_spike)  # update STP state, and get effective input
        current_inputs_shape = self._resolve_inputs_shape(pre_spike)
        effective_spike = effective_spike.view(current_inputs_shape)  # (batch_size, 1, 28, 28)
        # --------------------
        # Exc to Inh用 effective spike卷積
        # effective_input = self.exc_to_inh(effective_spike)

        # self.g_ampa.add_((effective_input - self.g_ampa / self.tau_ampa) * self.dt) # nine to one
        self.g_ampa.add_((effective_spike - self.g_ampa / self.tau_ampa) * self.dt) # one to one
        self.g_nmda.add_((self.g_ampa - self.g_nmda / self.tau_nmda) * self.dt)

        # 輸出總 excitatory conductance
        g_exc = self.alpha_inh * self.g_ampa + (1 - self.alpha_inh) * self.g_nmda

        return g_exc

    def reset(self):
        self.g_ampa.zero_()
        self.g_nmda.zero_()
        self.stp.reset()
    
    def build(self, pre_spike):
        self.stp.build(pre_spike)
        effective_spike = (pre_spike)
        current_inputs_shape = self._resolve_inputs_shape(pre_spike)
        effective_spike = effective_spike.view(current_inputs_shape)  # (batch_size, 1, 28, 28)
        effective_input = self.exc_to_inh(effective_spike)
        self.g_ampa = torch.zeros(effective_input.shape, device=self.device)
        self.g_nmda = torch.zeros(effective_input.shape, device=self.device)

        return self.g_nmda

# inputs IE synapse
class inputs_IE_Synapse(nn.Module):
    def __init__(self, pre_neurons, post_neurons, dt=1.0, 
                 tau_gaba=10.0, tau_adapt=100.0,
                 delta_a=0.1, 
                 device='cpu'):
        
        super().__init__()
        self.dt = dt
        self.device = device
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons

        # 時間常數
        self.tau_gaba = tau_gaba
        self.tau_adapt = tau_adapt
        self.delta_a = delta_a  # adaptation strength

        # 初始化狀態值
        # self.g_gaba = torch.zeros(post_neurons, device=device)
        # self.g_adapt = torch.zeros(post_neurons, device=device)
        self.g_gaba = None
        self.g_adapt = None

        # Inhibitory feedback → Excitatory (transpose convolution)
        self.inh_to_exc = nn.ConvTranspose2d(
            in_channels=1, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False # padding=1 保持輸出大小不變28 -> 28
        )
        self.inh_to_exc.weight.data.fill_(0.2)
        # self.inh_to_exc.weight.data.uniform_(0.19,0.21)

    def forward(self, pre_spike):
        # 更新 GABA conductance
        input_gaba = self.inh_to_exc(pre_spike)  # inhibitory inputs
        self.g_gaba += self.dt * (-self.g_gaba / self.tau_gaba + input_gaba)

        return self.g_gaba

    def update_g_adapt(self, post_spike):
        # print(post_spike.shape)
        # g_adapt = self.g_adapt
        self.g_adapt += self.dt * (-self.g_adapt / self.tau_adapt + self.delta_a * post_spike)
        return self.g_adapt

    def reset(self):
        self.g_gaba.zero_()
        self.g_adapt.zero_()
    
    def build(self, pre_spike):
        input_gaba = self.inh_to_exc(pre_spike)  # inhibitory inputs
        self.g_gaba = torch.zeros(input_gaba.shape, device=self.device)
        self.g_adapt = torch.zeros(input_gaba.shape, device=self.device)
        return self.g_gaba, self.g_adapt
