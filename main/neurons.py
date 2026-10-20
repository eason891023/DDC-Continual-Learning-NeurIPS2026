import torch
import torch.nn as nn
import os

class AdaptiveLIFNeuron(nn.Module):
    def __init__(self, dt=1.0, 
                 tau_m=20.0, tau_th=5.0,
                 u_rest=-70.0, u_exc=0.0, u_inh=-80.0,
                 theta_rest=-50.0, delta_theta=2.0,
                 device='cpu'):
        super().__init__()
        self.dt = dt
        self.device = device

        # membrane parameters
        self.tau_m = tau_m
        self.u_rest = u_rest
        self.u_exc = u_exc
        self.u_inh = u_inh

        # adaptive threshold parameters
        self.tau_th = tau_th
        self.theta_rest = theta_rest
        self.delta_theta = delta_theta

        # State variables（初始是None，lazy init）
        self.v = None
        self.theta = None
        self.spike = None

        # (optional) 可選的座標（如果要空間分布）
        self.coordinates = None

    def forward(self, g_exc, g_inh):
        # dv/dt update
        dv = (self.u_rest - self.v +
              g_exc * (self.u_exc - self.v) +
              g_inh * (self.u_inh - self.v)) / self.tau_m

        self.v += self.dt * dv

        # spike generation
        self.spike = (self.v >= self.theta).float()

        # if spike: reset potential and raise threshold
        self.v[self.spike.bool()] = self.u_rest
        # self.theta[self.spike.bool()] += self.delta_theta

        # # threshold decay for non-spiking neurons
        # no_spike_mask = (self.spike == 0)
        # dth = torch.zeros_like(self.theta)
        # dth[no_spike_mask] = (self.theta_rest - self.theta[no_spike_mask]) / self.tau_th
        # self.theta += self.dt * dth

        return self.spike

    def reset(self):
        self.v = None
        self.theta = None
        self.spike = None

    def build(self, g_exc, g_gaba):
        self.v = torch.full_like(g_exc, self.u_rest)
        self.theta = torch.full_like(g_exc, self.theta_rest)
        self.spike = torch.zeros_like(g_exc)
        return self.spike

# without g_a
class Inputs_AdaptiveLIFNeuronlayer(nn.Module):
    def __init__(self, inputs_size, dt=1.0, 
                 tau_m=20.0, tau_th=5.0,
                 u_rest=-70.0, u_exc=0.0, u_inh=-80.0,
                 theta_rest=-50.0, delta_theta=2.0,
                 device='cpu'):
        super().__init__()
        self.inputs_size = inputs_size
        self.dt = dt
        self.device = device

        # membrane parameters
        self.tau_m = tau_m
        self.u_rest = u_rest
        self.u_exc = u_exc
        self.u_inh = u_inh

        # adaptive threshold parameters
        self.tau_th = tau_th
        self.theta_rest = theta_rest
        self.delta_theta = delta_theta

        # State variables（初始是None，lazy init）
        self.v = None
        self.theta = None
        self.spike = None

    def forward(self, g_exc, g_inh):
        # dv/dt update
        dv = (self.u_rest - self.v + 
              g_exc * (self.u_exc - self.v) +
              g_inh * (self.u_inh - self.v)) / self.tau_m

        self.v += self.dt * dv

        # spike generation
        self.spike = (self.v >= self.theta).float()
        # print(f"spike:", self.spike[0][0][10][10:20])

        # if spike: reset potential and raise threshold
        self.v[self.spike.bool()] = self.u_rest
        # self.theta[self.spike.bool()] += self.delta_theta

        # # threshold decay
        no_spike_mask = (self.spike == 0)
        dth = torch.zeros_like(self.theta)
        dth[no_spike_mask] = (self.theta_rest - self.theta[no_spike_mask]) / self.tau_th
        self.theta += self.dt * dth

        return self.spike

    def reset(self):
        self.v.fill_(self.u_rest)
        self.theta.fill_(self.theta_rest)
        self.spike.zero_()

    def load_or_set_coordinates(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 確保資料夾存在

        if os.path.exists(path):
            self.coordinates = torch.load(path, map_location=self.device)
            print(f"Coordinates loaded from {path}")
        else:
            # 將 input neurons 均勻排列在 z=2 的平面上
            H, W = self.inputs_size[2], self.inputs_size[3]  # height, width
            x_vals = torch.linspace(-1, 1, W)
            y_vals = torch.linspace(1, -1, H)
            grid_y, grid_x = torch.meshgrid(y_vals, x_vals, indexing='ij')  # shape: [H, W]

            x_flat = grid_x.reshape(-1)
            y_flat = grid_y.reshape(-1)
            z_flat = torch.full_like(x_flat, 2.0)

            self.coordinates = torch.stack([x_flat, y_flat, z_flat], dim=1).to(self.device)  # (H*W, 3)
            torch.save(self.coordinates, path)
    
    def get_coordinates(self):
        return self.coordinates
    
    def show_coordinates(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates[:, 0].cpu(), self.coordinates[:, 1].cpu(), self.coordinates[:, 2].cpu())
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
    
    def build(self, g_exc, g_gaba):
        self.v = torch.full_like(g_exc, self.u_rest)
        self.theta = torch.full_like(g_exc, self.theta_rest)
        self.spike = torch.zeros_like(g_exc)
        return self.spike
    
    def get_threshold(self):
        return self.theta - self.u_rest
    
# without g_a
# class AdaptiveLIFNeuronlayer(nn.Module):
#     def __init__(self, num_neurons, dt=1.0, 
#                  tau_m=20.0, tau_th=5.0,
#                  u_rest=-70.0, u_exc=0.0, u_inh=-80.0,
#                  theta_rest=-50.0, delta_theta=2.0,
#                  device='cpu'):
#         super().__init__()
#         self.num_neurons = num_neurons
#         self.dt = dt
#         self.device = device

#         # membrane parameters
#         self.tau_m = tau_m
#         self.u_rest = u_rest
#         self.u_exc = u_exc
#         self.u_inh = u_inh

#         # adaptive threshold parameters
#         self.tau_th = tau_th
#         self.theta_rest = theta_rest
#         self.delta_theta = delta_theta

#         # State variables（初始是None，lazy init）
#         self.v = None
#         self.theta = None
#         self.spike = None

#         # Initialize the coordinates for visualization
#         self.coordinates = torch.zeros((num_neurons, 3), device=device)

#     def forward(self, g_exc, g_inh):
#         # print(f"g_exc: {g_exc.mean()}, g_inh: {g_inh.mean()}")
#         # dv/dt update
#         dv = (self.u_rest - self.v + 
#               g_exc * (self.u_exc - self.v) +
#               (g_inh) * (self.u_inh - self.v)) / self.tau_m

#         # print("membrane potential: ", self.v[0][:10])
#         # print("g_exc: ", g_exc[0][:10])
#         # print("g_inh: ", g_inh[0][:10])
#         # print("effective g_exc: ", (g_exc * (self.u_exc - self.v))[0][:10])
#         # print("effective g_inh: ", (g_inh * (self.u_inh - self.v))[0][:10])
#         # print("dv: ", dv[0][:10])
#         # print("threshold: ", self.theta[0][:10])
#         self.v += self.dt * dv

#         # spike generation
#         self.spike = (self.v >= self.theta).float()

#         # if spike: reset potential and raise threshold
#         self.v[self.spike.bool()] = self.u_rest
#         self.theta[self.spike.bool()] += self.delta_theta

#         # print(f"self.spike: {self.spike[0:5]}")
#         # threshold decay
#         no_spike_mask = (self.spike == 0)
#         dth = torch.zeros_like(self.theta)
#         dth[no_spike_mask] = (self.theta_rest - self.theta[no_spike_mask]) / self.tau_th
#         self.theta += self.dt * dth

#         return self.spike

#     def reset(self):
#         self.v.fill_(self.u_rest)
#         self.theta.fill_(self.theta_rest)
#         self.spike.zero_()

#     def load_or_set_coordinates(self, path, radius = 1.0):
#         if os.path.exists(path):
#             self.coordinates = torch.load(path, map_location=self.device)
#             print(f"Coordinates loaded from {path}")
#         else:
#             random_coordinates = torch.rand((self.num_neurons, 3), device=self.device) * 2 - 1  # [-1, 1]
#             random_coordinates /= random_coordinates.norm(dim=1, keepdim=True)
#             scale = torch.rand(self.num_neurons, device=self.device).pow(1/3) * radius  # 讓它們不只在表面
#             random_coordinates = random_coordinates * scale[:, None]
#             self.coordinates = random_coordinates
#             torch.save(self.coordinates, path)
#         # print(f"Coordinates set to {self.coordinates}")
    
#     def get_coordinates(self):
#         return self.coordinates
    
#     def show_coordinates(self):
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(self.coordinates[:, 0].cpu(), self.coordinates[:, 1].cpu(), self.coordinates[:, 2].cpu())
#         ax.set_xlabel('X axis')
#         ax.set_ylabel('Y axis')
#         ax.set_zlabel('Z axis')
#         plt.show()
    
#     def build(self, g_exc, g_gaba):
#         self.v = torch.full_like(g_exc, self.u_rest)
#         self.theta = torch.full_like(g_exc, self.theta_rest)
#         self.spike = torch.zeros_like(g_exc)
#         return self.spike
        
###########################################################################
# # Adaptive LIF Neuron with dynamic threshold and spatial coordinates    #
# # Lazy initialization of state variables                #               #
# With refractory period and coordinates for visualization #
###########################################################################

# class AdaptiveLIFNeuron(nn.Module):
#     def __init__(self, dt=1.0, 
#                  tau_m=20.0, tau_th=5.0,
#                  u_rest=-70.0, u_exc=0.0, u_inh=-80.0,
#                  theta_rest=-50.0, delta_theta=2.0,
#                  refractory_period=2.0,
#                  device='cpu'):
#         super().__init__()
#         self.dt = dt
#         self.device = device
#         self.refractory_steps = int(refractory_period / dt)
#         self.refractory_counter = None  # 每個神經元的剩餘不應期步數

#         # membrane parameters
#         self.tau_m = tau_m
#         self.u_rest = u_rest
#         self.u_exc = u_exc
#         self.u_inh = u_inh

#         # adaptive threshold parameters
#         self.tau_th = tau_th
#         self.theta_rest = theta_rest
#         self.delta_theta = delta_theta

#         # State variables（初始是None，lazy init）
#         self.v = None
#         self.theta = None
#         self.spike = None

#         # (optional) 可選的座標（如果要空間分布）
#         self.coordinates = None

#     def forward(self, g_exc, g_inh):
#         not_refractory = self.refractory_counter <= 0

#         dv = (self.u_rest - self.v +
#               g_exc * (self.u_exc - self.v) +
#               g_inh * (self.u_inh - self.v)) / self.tau_m

#         self.v[not_refractory] += self.dt * dv[not_refractory]

#         self.spike = torch.zeros_like(self.v)
#         spiking_neurons = (self.v >= self.theta) & not_refractory
#         self.spike[spiking_neurons] = 1.0

#         self.v[spiking_neurons] = self.u_rest
#         self.theta[spiking_neurons] += self.delta_theta

#         # 閾值恢復
#         no_spike_mask = (self.spike == 0)
#         dth = torch.zeros_like(self.theta)
#         dth[no_spike_mask] = (self.theta_rest - self.theta[no_spike_mask]) / self.tau_th
#         self.theta += self.dt * dth

#         # 更新不應期計數器
#         self.refractory_counter[spiking_neurons] = self.refractory_steps
#         self.refractory_counter -= 1

#         return self.spike

#     def reset(self):
#         self.v = None
#         self.theta = None
#         self.spike = None

#     def build(self, g_exc, g_inh):
#         self.v = torch.full_like(g_exc, self.u_rest)
#         self.theta = torch.full_like(g_exc, self.theta_rest)
#         self.spike = torch.zeros_like(g_exc)
#         self.refractory_counter = torch.zeros_like(g_exc)
#         return self.spike

# # without g_a
# class Inputs_AdaptiveLIFNeuronlayer(nn.Module):
#     def __init__(self, inputs_size, dt=1.0, 
#                  tau_m=20.0, tau_th=5.0,
#                  u_rest=-70.0, u_exc=0.0, u_inh=-80.0,
#                  theta_rest=-50.0, delta_theta=2.0,
#                  refractory_period=2.0,
#                  device='cpu'):
#         super().__init__()
#         self.inputs_size = inputs_size
#         self.dt = dt
#         self.device = device
#         self.refractory_steps = int(refractory_period / dt)
#         self.refractory_counter = None  # 每個神經元的剩餘不應期步數

#         # membrane parameters
#         self.tau_m = tau_m
#         self.u_rest = u_rest
#         self.u_exc = u_exc
#         self.u_inh = u_inh

#         # adaptive threshold parameters
#         self.tau_th = tau_th
#         self.theta_rest = theta_rest
#         self.delta_theta = delta_theta

#         # State variables（初始是None，lazy init）
#         self.v = None
#         self.theta = None
#         self.spike = None

#     def forward(self, g_exc, g_inh):
#         not_refractory = self.refractory_counter <= 0

#         dv = (self.u_rest - self.v +
#               g_exc * (self.u_exc - self.v) +
#               g_inh * (self.u_inh - self.v)) / self.tau_m

#         self.v[not_refractory] += self.dt * dv[not_refractory]

#         self.spike = torch.zeros_like(self.v)
#         spiking_neurons = (self.v >= self.theta) & not_refractory
#         self.spike[spiking_neurons] = 1.0

#         self.v[spiking_neurons] = self.u_rest
#         self.theta[spiking_neurons] += self.delta_theta

#         # 閾值恢復
#         no_spike_mask = (self.spike == 0)
#         dth = torch.zeros_like(self.theta)
#         dth[no_spike_mask] = (self.theta_rest - self.theta[no_spike_mask]) / self.tau_th
#         self.theta += self.dt * dth

#         # 更新不應期計數器
#         self.refractory_counter[spiking_neurons] = self.refractory_steps
#         self.refractory_counter -= 1

#         return self.spike


#     def reset(self):
#         self.v.fill_(self.u_rest)
#         self.theta.fill_(self.theta_rest)
#         self.spike.zero_()

#     def load_or_set_coordinates(self, path):
#         if os.path.exists(path):
#             self.coordinates = torch.load(path, map_location=self.device)
#             print(f"Coordinates loaded from {path}")
#         else:
#             """
#             將 input neurons 均勻排列在 z=2 的平面上。
#             x 從 -1 到 1，y 從 1 到 -1。
#             """
#             H, W = self.inputs_size[2], self.inputs_size[3]  # height, width
#             x_vals = torch.linspace(-1, 1, W)
#             y_vals = torch.linspace(1, -1, H)
#             grid_y, grid_x = torch.meshgrid(y_vals, x_vals, indexing='ij')  # shape: [H, W]

#             x_flat = grid_x.reshape(-1)
#             y_flat = grid_y.reshape(-1)
#             z_flat = torch.full_like(x_flat, 2.0)

#             self.coordinates = torch.stack([x_flat, y_flat, z_flat], dim=1).to(self.device)  # (H*W, 3)
#             torch.save(self.coordinates, path)
#         # print(f"Coordinates set to {self.coordinates}")
    
#     def get_coordinates(self):
#         return self.coordinates
    
#     def show_coordinates(self):
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(self.coordinates[:, 0].cpu(), self.coordinates[:, 1].cpu(), self.coordinates[:, 2].cpu())
#         ax.set_xlabel('X axis')
#         ax.set_ylabel('Y axis')
#         ax.set_zlabel('Z axis')
#         plt.show()
    
#     def build(self, g_exc, g_gaba):
#         self.v = torch.full_like(g_exc, self.u_rest)
#         self.theta = torch.full_like(g_exc, self.theta_rest)
#         self.spike = torch.zeros_like(g_exc)
#         self.refractory_counter = torch.zeros_like(g_exc)
#         return self.spike
    
#     def get_threshold(self):
#         return self.theta - self.u_rest
    
# # without g_a
class AdaptiveLIFNeuronlayer(nn.Module):
    def __init__(self, num_neurons, dt=1.0, 
                 tau_m=20.0, tau_th=5.0,
                 u_rest=-70.0, u_exc=0.0, u_inh=-80.0,
                 theta_rest=-50.0,
                 delta_theta=2.0,
                 refractory_period=0.0,
                 shape = 'sphere',
                 device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.device = device
        self.refractory_steps = int(refractory_period / dt)
        self.refractory_counter = None  # 每個神經元的剩餘不應期步數

        # membrane parameters
        self.tau_m = tau_m
        self.u_rest = u_rest
        self.u_exc = u_exc
        self.u_inh = u_inh

        # adaptive threshold parameters
        self.tau_th = tau_th
        self.theta_rest = theta_rest
        self.delta_theta = delta_theta

        # State variables（初始是None，lazy init）
        self.v = None
        self.theta = None
        self.spike = None
        
        self.shape = shape

        # Initialize the coordinates for visualization
        self.coordinates = torch.zeros((num_neurons, 3), device=device)

    def forward(self, g_exc, g_inh):
        not_refractory = self.refractory_counter <= 0

        dv = (self.u_rest - self.v +
              g_exc * (self.u_exc - self.v) +
              g_inh * (self.u_inh - self.v)) / self.tau_m

        self.v[not_refractory] += self.dt * dv[not_refractory]

        self.spike = torch.zeros_like(self.v)
        spiking_neurons = (self.v >= self.theta) & not_refractory
        self.spike[spiking_neurons] = 1.0

        self.v[spiking_neurons] = self.u_rest
        self.theta[spiking_neurons] += self.delta_theta
        # self.theta[spiking_neurons] = 100  # 直接設為100，避免閾值過高導致後續無法產生spike # 50%而已

        # 閾值恢復
        no_spike_mask = (self.spike == 0)
        dth = torch.zeros_like(self.theta)
        dth[no_spike_mask] = (self.theta_rest - self.theta[no_spike_mask]) / self.tau_th
        self.theta += self.dt * dth

        # 更新不應期計數器
        self.refractory_counter[spiking_neurons] = self.refractory_steps
        self.refractory_counter -= 1

        return self.spike


    def reset(self):
        self.v.fill_(self.u_rest)
        self.theta.fill_(self.theta_rest)
        self.spike.zero_()

    def load_or_set_coordinates(self, path, radius = 1.0, shape = 'sphere'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if shape == 'sphere':
            if os.path.exists(path):
                self.coordinates = torch.load(path, map_location=self.device)
                print(f"Coordinates loaded from {path}")
            else:
                random_coordinates = torch.rand((self.num_neurons, 3), device=self.device) * 2 - 1  # [-1, 1]
                random_coordinates /= random_coordinates.norm(dim=1, keepdim=True)
                scale = torch.rand(self.num_neurons, device=self.device).pow(1/3) * radius  # 讓它們不只在表面
                random_coordinates = random_coordinates * scale[:, None]
                self.coordinates = random_coordinates
                torch.save(self.coordinates, path)
        # ellipse
        else:
            if os.path.exists(path):
                self.coordinates = torch.load(path, map_location=self.device)
                print(f"Coordinates loaded from {path}")
            else:
                # 均勻撒點於橢球體內部
                N = self.num_neurons
                coords = []

                while len(coords) < N:
                    # 在 x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [-0.3, 0.3] 的立方體中生成隨機點
                    point = torch.rand(3, device=self.device)
                    point[0] = point[0] * 2 - 1        # x ∈ [-1, 1]
                    point[1] = point[1] * 2 - 1        # y ∈ [-1, 1]
                    point[2] = point[2] * 0.6 - 0.3    # z ∈ [-0.3, 0.3]

                    # 檢查是否在橢球體內部：x² + y² + (z/0.3)² ≤ 1
                    if (point[0]**2 + point[1]**2 + (point[2] / 0.3)**2 <= 1):
                        coords.append(point)

                self.coordinates = torch.stack(coords)
                torch.save(self.coordinates, path)
                print(f"{N} neurons randomly placed in elliptical volume and saved to {path}")

        # print(f"Coordinates set to {self.coordinates}")
    
    def get_coordinates(self):
        return self.coordinates
    
    # def show_coordinates(self):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(self.coordinates[:, 0].cpu(), self.coordinates[:, 1].cpu(), self.coordinates[:, 2].cpu())
    #     ax.set_xlabel('X axis')
    #     ax.set_ylabel('Y axis')
    #     ax.set_zlabel('Z axis')
    #     plt.show()

    def show_coordinates(self, inactive_indices=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        coords = self.coordinates.cpu()

        # 所有神經元：灰色小點（背景）
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                c='gray', alpha=0.1, s=8, label='All Neurons')

        # 沒發 spike 的神經元：紅色半透明點，略大一點
        if inactive_indices is not None and len(inactive_indices) > 0:
            inactive_coords = coords[inactive_indices]
            ax.scatter(inactive_coords[:, 0], inactive_coords[:, 1], inactive_coords[:, 2],
                    c='red', alpha=0.4, s=18, label='No Spike')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()
        plt.title("Neuron Coordinates (Red = No Spike, Transparent)")
        plt.show()
    
    def build(self, g_exc, g_gaba):
        self.v = torch.full_like(g_exc, self.u_rest)
        self.theta = torch.full_like(g_exc, self.theta_rest)
        self.spike = torch.zeros_like(g_exc)
        self.refractory_counter = torch.zeros_like(g_exc)
        return self.spike