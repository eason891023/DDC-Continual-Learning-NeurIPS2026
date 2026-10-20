import torch
import torch.nn as nn
from .neurons import *
from .synapse import *
from tqdm import tqdm
import torch.nn.functional as F
from .synapse_plasticity import *
import math
import numpy as np
import json
import os
@torch.no_grad()
class Brain_Like_RSNN(nn.Module):
    def __init__(self, inputs_size, exc_neurons, inh_neurons, sigma=1.0, dt=0.1, mode='random', input_mode='spatial', target_mean=0.05, seed=None, device='cpu'):
        super().__init__()
        self.device = device
        self.dt = dt
        self.seed = seed

        # Excitatory and Inhibitory neurons
        self.exc_neurons = exc_neurons
        self.inh_neurons = inh_neurons
        self.inputs_size = inputs_size

        # mode = "distance" or "random" or FC
        self.mode = mode
        self.sigma = sigma      # 決定參數的
        self.target_mean = target_mean
        self.input_mode = input_mode  # 'spatial'
        self.shape = 'sphere'
        # shape = 'ellipse'

        # 將28x28 的inputs padding成30x30
        # # conv inputs
        # self.padding = nn.ConstantPad2d(1, 0)

        # Inputs → Excitatory：用Conv2d（3x3，stride=1，padding=0）
        # self.input_to_exc = nn.Conv2d(
        #     in_channels=1, out_channels=1,
        #     kernel_size=3, stride=1, padding=0, bias=False, device=device
        # )
        # self.input_to_exc.weight.data.fill_(1.0)  # 簡單設成全部加總

        size_str = "x".join(map(str, self.inputs_size))
        # Seed-specific coordinate paths (RSNN neurons are random; input neurons are deterministic grid)
        seed_dir = f"seed_{seed}" if seed is not None else "default"
        # self.ASF = AdaptiveSynapticFilter()
        # inputs layers neurons
        self.inputs_Exc_neurons = Inputs_AdaptiveLIFNeuronlayer(inputs_size, dt=dt, device=device)
        self.inputs_Exc_neurons.load_or_set_coordinates(f"coordinates/{size_str}/inputs_Exc_coords.pt")  # deterministic grid, no seed needed
        self.inputs_Exc_coordinates = self.inputs_Exc_neurons.get_coordinates()
        self.inputs_Inh_neurons = AdaptiveLIFNeuron(dt=dt, device=device)

        self.inputs_Exc_spikes = torch.zeros(inputs_size, device=device)

        self.inh_neurons_inh_inputs = torch.zeros(inputs_size, device=device)
        self.inputs_gaba = torch.zeros(inputs_size, device=device)
        self.inputs_adapt = torch.zeros(inputs_size, device=device)
        self.inputs_Inh = self.inputs_gaba + self.inputs_adapt

        # inputs layers synapse
        self.inputs_EI_synapase = inputs_EI_Synapse(inputs_size, exc_neurons, dt=dt, device=device, inputs_shape=inputs_size)
         # inputs inhibitory synapse to inputs excitatory neurons
        self.inputs_IE_synapase = inputs_IE_Synapse(inputs_size, inh_neurons, dt=dt, device=device)

        # RSNN excitatory
        self.RSNN_Exc_neurons = AdaptiveLIFNeuronlayer(exc_neurons, dt=dt, device=device, refractory_period=1)
        self.RSNN_Exc_neurons.load_or_set_coordinates(f"coordinates/{self.shape}/{seed_dir}/RSNN_Exc_coords.pt", shape = self.shape)
        self.RSNN_Exc_coordinates = self.RSNN_Exc_neurons.get_coordinates()
        # RSNN inhibitory
        self.RSNN_Inh_neurons = AdaptiveLIFNeuronlayer(inh_neurons, dt=dt, device=device, refractory_period=0.5)
        self.RSNN_Inh_neurons.load_or_set_coordinates(f"coordinates/{self.shape}/{seed_dir}/RSNN_Inh_coords.pt", shape = self.shape)
        self.RSNN_Inh_coordinates = self.RSNN_Inh_neurons.get_coordinates()

        # inputs layers to RSNN excitatory neurons
        self.inputs2RSNN_EE_synapase = inputs_2_RSNN_EESynapse(inputs_size, exc_neurons,
                                                               Inputs_coordinates = self.inputs_Exc_coordinates, RSNN_coordinates = self.RSNN_Exc_coordinates,
                                                               dt=dt, device=device, shape=self.shape, inputs_shape=inputs_size,
                                                               input_mode=self.input_mode, seed=seed)

        # RSNN synapse
        self.RSNN_EE_synapase = EE_Synapse(exc_neurons, exc_neurons, RSNN_coordinates = self.RSNN_Exc_coordinates, dt=dt, device=device, mode = self.mode, sigma=sigma, shape=self.shape, target_mean=self.target_mean, seed=seed)
        self.RSNN_EI_synapase = EI_Synapse(exc_neurons, inh_neurons, RSNN_Exc_coordinates = self.RSNN_Exc_coordinates, RSNN_Inh_coordinates = self.RSNN_Inh_coordinates, dt=dt, device=device, mode = self.mode , sigma=self.sigma, shape=self.shape, target_mean=self.target_mean, seed=seed)
        self.RSNN_IE_synapase = IE_Synapse(inh_neurons, exc_neurons, RSNN_Exc_coordinates = self.RSNN_Exc_coordinates, RSNN_Inh_coordinates = self.RSNN_Inh_coordinates, dt=dt, device=device, mode = self.mode , sigma=self.sigma, shape=self.shape, target_mean=self.target_mean, seed=seed)
        self.RSNN_II_synapase = II_Synapse(inh_neurons, inh_neurons, RSNN_coordinates = self.RSNN_Inh_coordinates, dt=dt, device=device, mode = self.mode , sigma=sigma, shape=self.shape, target_mean=self.target_mean, seed=seed)

        # spikes tensor
        self.inputs_Exc_spikes = None
        self.inputs_Inh_spikes = None
        self.RSNN_Exc_pre_spikes = None
        self.RSNN_Exc_post_spikes = None
        self.RSNN_Inh_pre_spikes = None
        self.RSNN_Inh_post_spikes = None


        # plasticity control
        self.plasticity_enabled = True

        # for testing
        self.T_burn_in = 5 * 1000
        self.T_training_on = 1000
        self.T_training_off = 2000
        self.background_noise_freq = 3

    def forward(self, inputs_signal):
        # inputs_signal: (batch_size, 1, 28, 28)
        # inputs_signal = self.padding(inputs_signal)  # 變成 (batch, 1, 30, 30)
        # inputs_signal = self.input_to_exc(inputs_signal)  # (batch, 1, 28, 28)
        # Vision inputs Layer
        # inputs exc sig = inputs signals + inputs inh signals
        # inputs_signal = inputs_signal * 15 # self.inputs_Exc_neurons.get_threshold() # inputs signals
        inputs_signal = inputs_signal # self.inputs_Exc_neurons.get_threshold() # inputs signals
        self.inputs_Exc_spikes = self.inputs_Exc_neurons(inputs_signal, self.inputs_Inh) # inputs excitatory neurons
        self.inputs_adapt = self.inputs_IE_synapase.update_g_adapt(self.inputs_Exc_spikes)
        # print(self.inputs_Exc_spikes)
        # inputs excitatory synapse to inputs inhibitory neurons
        inputs_Inh_neurons_g_exc = self.inputs_EI_synapase(self.inputs_Exc_spikes)

        self.inh_neurons_inh_inputs = torch.zeros(inputs_Inh_neurons_g_exc.shape, device=self.device) # always 0

        # inputs inh signals = inputs exc signals
        self.inputs_Inh_spikes = self.inputs_Inh_neurons(inputs_Inh_neurons_g_exc, self.inh_neurons_inh_inputs) # inputs inhibitory neurons
        # print(self.inputs_Inh_spikes)
        # inputs inhibitory synapse to inputs excitatory neurons 
        self.inputs_gaba = self.inputs_IE_synapase(self.inputs_Inh_spikes) # Inhibit inputs Exc Neurons at next time step
        # print(self.inputs_Inh_spikes)

        # self.inputs_Exc_spikes = self.inputs_Exc_neurons(inputs_signal, self.inh_neurons_inh_inputs) # inputs excitatory neurons
        # inputs excitatory synapse to RSNN excitatory neurons
        inputs2RSNN_g_exc = self.inputs2RSNN_EE_synapase(self.inputs_Exc_spikes)
        RSNN_Exc_g_exc = self.RSNN_EE_synapase(self.RSNN_Exc_pre_spikes)
        RSNN_Exc_g_gaba = self.RSNN_IE_synapase(self.RSNN_Inh_pre_spikes)

        # RSNN excitatory neurons
        self.RSNN_Exc_post_spikes = self.RSNN_Exc_neurons(inputs2RSNN_g_exc + RSNN_Exc_g_exc, RSNN_Exc_g_gaba + self.RSNN_IE_synapase.g_adapt)
        # print("RSNN_Exc_post_spikes:", self.RSNN_Exc_post_spikes[0][:10])
        # print(self.RSNN_Exc_post_spikes[0][:10])
        self.RSNN_IE_synapase.update_g_adapt(self.RSNN_Exc_post_spikes)
        # print(self.RSNN_Exc_post_spikes)

        RSNN_Inh_g_exc = self.RSNN_EI_synapase(self.RSNN_Exc_pre_spikes)
        RSNN_Inh_g_gaba = self.RSNN_II_synapase(self.RSNN_Inh_pre_spikes)

        # RSNN inhibitory neurons
        self.RSNN_Inh_post_spikes = self.RSNN_Inh_neurons(RSNN_Inh_g_exc, (RSNN_Inh_g_gaba + self.RSNN_II_synapase.g_adapt))
        self.RSNN_II_synapase.update_g_adapt(self.RSNN_Inh_post_spikes)

        # print("inputs exc spike:", self.inputs_Exc_spikes)
        # print("exc spike:", self.RSNN_Exc_post_spikes)
        # print("inh spike:", self.RSNN_Inh_post_spikes)

        return self.RSNN_Exc_post_spikes, self.RSNN_Inh_post_spikes
    
    def enable_plasticity(self):
        self.plasticity_enabled = True

    def disable_plasticity(self):
        self.plasticity_enabled = False

    def step(self):
        self.update()
        self.RSNN_Exc_pre_spikes.copy_(self.RSNN_Exc_post_spikes.detach())
        self.RSNN_Inh_pre_spikes.copy_(self.RSNN_Inh_post_spikes.detach())

    def update(self):
        if not self.plasticity_enabled:
            return
        # updata inputs 2 RSNN synapse
        self.inputs2RSNN_EE_synapase.update(self.inputs_Exc_spikes, self.RSNN_Exc_post_spikes) # update inputs 2 RSNN EE synapse

        # # update RSNN synapse
        self.RSNN_EE_synapase.update(self.RSNN_Exc_pre_spikes, self.RSNN_Exc_post_spikes) # update RSNN EE synapse
        ## EI synapse update state at its forward pass
        self.RSNN_IE_synapase.update(self.RSNN_Inh_pre_spikes, self.RSNN_Exc_pre_spikes) # update RSNN IE synapse

    def burn_in(self, train_loader, T_on=2000, T_off=3000):
        """
        Burn-in period: feed one random MNIST image for T_on steps,
        then silence for T_off steps (zero input).
        Excitatory plasticity (LTP) is OFF; inhibitory STDP stays ON
        so the homeostatic regulation can stabilise network activity.
        """
        # Pick a random image from the loader
        all_data = list(train_loader)
        rand_idx = torch.randint(0, len(all_data), (1,)).item()
        burn_img, burn_label = all_data[rand_idx]
        burn_img = burn_img.to(self.device)
        zero_input = torch.zeros_like(burn_img)

        print(f"Starting burn-in: ON={T_on} steps (label={burn_label.item()}), OFF={T_off} steps ...")
        with torch.no_grad():
            for t in range(T_on):
                self.forward(burn_img)
                # Only inhibitory STDP (homeostatic), skip excitatory LTP
                self.RSNN_IE_synapase.update(self.RSNN_Inh_pre_spikes, self.RSNN_Exc_pre_spikes)
                self.RSNN_Exc_pre_spikes.copy_(self.RSNN_Exc_post_spikes.detach())
                self.RSNN_Inh_pre_spikes.copy_(self.RSNN_Inh_post_spikes.detach())
            for t in range(T_off):
                self.forward(zero_input)
                self.RSNN_IE_synapase.update(self.RSNN_Inh_pre_spikes, self.RSNN_Exc_pre_spikes)
                self.RSNN_Exc_pre_spikes.copy_(self.RSNN_Exc_post_spikes.detach())
                self.RSNN_Inh_pre_spikes.copy_(self.RSNN_Inh_post_spikes.detach())

        print(f"Burn-in complete. Exc spikes last step: {self.RSNN_Exc_post_spikes.sum().item():.0f}, "
              f"Inh spikes: {self.RSNN_Inh_post_spikes.sum().item():.0f}")
    
    def build(self, inputs_signal):
        # inputs_signal: (batch_size, 1, 28, 28)
        # inputs_signal = self.padding(inputs_signal)  # 變成 (batch, 1, 30, 30)
        # inputs_signal = self.input_to_exc(inputs_signal)  # (batch, 1, 28, 28)
        # Vision inputs Layer
        # inputs exc sig = inputs signals + inputs inh signals
        # self.inputs_Exc_spikes = self.inputs_Exc_neurons(inputs_signal, self.inputs_gaba, self.inputs_adapt)
        self.inputs_Exc_spikes = self.inputs_Exc_neurons.build(inputs_signal, self.inh_neurons_inh_inputs) # inputs excitatory neurons
        # inputs excitatory synapse to inputs inhibitory neurons
        inputs_Inh_neurons_g_exc = self.inputs_EI_synapase.build(self.inputs_Exc_spikes)

        # self.RSNN_Exc_pre_spikes.shape = [batch_size, exc_neurons]
        self.RSNN_Exc_pre_spikes = torch.zeros((inputs_Inh_neurons_g_exc.shape[0], self.exc_neurons), device=self.device)
        self.RSNN_Inh_pre_spikes = torch.zeros((inputs_Inh_neurons_g_exc.shape[0], self.inh_neurons), device=self.device)
        self.RSNN_Exc_post_spikes = torch.zeros((inputs_Inh_neurons_g_exc.shape[0], self.exc_neurons), device=self.device)
        self.RSNN_Inh_post_spikes = torch.zeros((inputs_Inh_neurons_g_exc.shape[0], self.inh_neurons), device=self.device)
        # print(inputs_Inh_neurons_g_exc)
        # inputs inh signals = inputs exc signals
        self.inputs_Inh_spikes = self.inputs_Inh_neurons.build(inputs_Inh_neurons_g_exc, self.inputs_Inh) # inputs inhibitory neurons
        # inputs inhibitory synapse to inputs excitatory neurons 
        self.inputs_gaba = self.inputs_IE_synapase.build(self.inputs_Inh_spikes) # Inhibit inputs Exc Neurons at next time step
        # print(self.inputs_Inh_spikes)

        # inputs excitatory synapse to RSNN excitatory neurons
        inputs2RSNN_g_exc = self.inputs2RSNN_EE_synapase.build(self.inputs_Exc_spikes, self.RSNN_Exc_post_spikes)
        RSNN_Exc_g_exc = self.RSNN_EE_synapase.build(self.RSNN_Exc_pre_spikes, self.RSNN_Exc_post_spikes)
        RSNN_Exc_g_gaba = self.RSNN_IE_synapase.build(self.RSNN_Inh_pre_spikes, self.RSNN_Exc_pre_spikes)

        # RSNN excitatory neurons
        self.RSNN_Exc_post_spikes = self.RSNN_Exc_neurons.build(inputs2RSNN_g_exc + RSNN_Exc_g_exc, RSNN_Exc_g_gaba)

        RSNN_Inh_g_exc = self.RSNN_EI_synapase.build(self.RSNN_Exc_pre_spikes)
        RSNN_Inh_g_gaba = self.RSNN_II_synapase.build(self.RSNN_Inh_pre_spikes)

        # RSNN inhibitory neurons
        self.RSNN_Inh_post_spikes = self.RSNN_Inh_neurons.build(RSNN_Inh_g_exc, RSNN_Inh_g_gaba)
    
    def plt_coordinates(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.inputs_Exc_coordinates[:, 0].cpu().numpy(),
                self.inputs_Exc_coordinates[:, 1].cpu().numpy(),
                self.inputs_Exc_coordinates[:, 2].cpu().numpy(),
                c='blue', s=1, label='Inputs Exc Neurons')

        ax.scatter(self.RSNN_Exc_coordinates[:, 0].cpu().numpy(),
                self.RSNN_Exc_coordinates[:, 1].cpu().numpy(),
                self.RSNN_Exc_coordinates[:, 2].cpu().numpy(),
                c='red', s=1, label='RSNN Exc Neurons')

        ax.scatter(self.RSNN_Inh_coordinates[:, 0].cpu().numpy(),
                self.RSNN_Inh_coordinates[:, 1].cpu().numpy(),
                self.RSNN_Inh_coordinates[:, 2].cpu().numpy(),
                c='green', s=1, label='RSNN Inh Neurons')

        ax.set_title('Neuron Coordinates (3D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    
    def save_all_synapse_weights(self, save_dir="weights", tag="init"):
        """
        存下所有Synapse中的weights
        Args:
            save_dir (str): 儲存資料夾
            tag (str): 檔名標籤 (例：'init' 或 'trained')
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        size_str = "x".join(map(str, self.inputs_size))
        input_weights_dir = os.path.join(save_dir, size_str)
        if not os.path.exists(input_weights_dir):
            os.makedirs(input_weights_dir)
        flat_input_weight_path = os.path.join(save_dir, f"inputs2RSNN_EE_weights_{tag}.pt")
        nested_input_weight_path = os.path.join(input_weights_dir, f"inputs2RSNN_EE_weights_{tag}.pt")

        # 存 Inputs → RSNN Exc Synapse weights
        torch.save(self.inputs2RSNN_EE_synapase.inputs_EE_weights.detach().cpu(), 
                flat_input_weight_path)
        torch.save(self.inputs2RSNN_EE_synapase.inputs_EE_weights.detach().cpu(), 
                nested_input_weight_path)

        # 存 RSNN Exc → Exc Synapse weights
        torch.save(self.RSNN_EE_synapase.RSNN_EE_weights.detach().cpu(), 
                os.path.join(save_dir, f"RSNN_EE_weights_{tag}.pt"))

        # 存 RSNN Exc → Inh Synapse weights
        torch.save(self.RSNN_EI_synapase.RSNN_EI_weights.detach().cpu(), 
                os.path.join(save_dir, f"RSNN_EI_weights_{tag}.pt"))

        # 存 RSNN Inh → Exc Synapse weights
        torch.save(self.RSNN_IE_synapase.IE_weights.detach().cpu(), 
                os.path.join(save_dir, f"RSNN_IE_weights_{tag}.pt"))

        # 存 RSNN Inh → Inh Synapse weights
        torch.save(self.RSNN_II_synapase.II_weights.detach().cpu(), 
                os.path.join(save_dir, f"RSNN_II_weights_{tag}.pt"))

        print(f"All synapse weights saved to {save_dir} (tag={tag})")

    
    def load_all_synapse_weights(self, save_dir="weights", tag="init"):
        """
        從資料夾讀取所有Synapse中的weights
        Args:
            save_dir (str): 儲存資料夾
            tag (str): 檔名標籤 (例：'init' 或 'trained')
        """
        size_str = "x".join(map(str, self.inputs_size))
        input_weight_candidates = [
            os.path.join(save_dir, f"inputs2RSNN_EE_weights_{tag}.pt"),
            os.path.join(save_dir, size_str, f"inputs2RSNN_EE_weights_{tag}.pt"),
            os.path.join(save_dir, f"{self.inputs_size}", f"inputs2RSNN_EE_weights_{tag}.pt"),
        ]
        input_weight_path = next((path for path in input_weight_candidates if os.path.exists(path)), None)
        if input_weight_path is None:
            raise FileNotFoundError(
                f"Could not find inputs2RSNN_EE weights for tag='{tag}' under {save_dir}. "
                f"Tried: {input_weight_candidates}"
            )

        # 讀 Inputs → RSNN Exc Synapse weights
        self.inputs2RSNN_EE_synapase.inputs_EE_weights.data.copy_(
            torch.load(input_weight_path, map_location=self.device)
        )

        # 讀 RSNN Exc → Exc Synapse weights
        self.RSNN_EE_synapase.RSNN_EE_weights.data.copy_(
            torch.load(os.path.join(save_dir, f"RSNN_EE_weights_{tag}.pt"), map_location=self.device)
        )

        # 讀 RSNN Exc → Inh Synapse weights
        self.RSNN_EI_synapase.RSNN_EI_weights.data.copy_(
            torch.load(os.path.join(save_dir, f"RSNN_EI_weights_{tag}.pt"), map_location=self.device)
        )

        # 讀 RSNN Inh → Exc Synapse weights
        self.RSNN_IE_synapase.IE_weights.data.copy_(
            torch.load(os.path.join(save_dir, f"RSNN_IE_weights_{tag}.pt"), map_location=self.device)
        )

        # 讀 RSNN Inh → Inh Synapse weights
        self.RSNN_II_synapase.II_weights.data.copy_(
            torch.load(os.path.join(save_dir, f"RSNN_II_weights_{tag}.pt"), map_location=self.device)
        )

        print(f"All synapse weights loaded from {save_dir} (tag={tag})")

    def save_checkpoint(self, save_dir, tag="trained"):
        """
        Save complete model checkpoint including weights, w_tilde, masks, and coordinates.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. All synapse weights
        self.save_all_synapse_weights(save_dir=save_dir, tag=tag)

        # 2. w_tilde (heterosynaptic reference weights — critical slow variable)
        torch.save(self.RSNN_EE_synapase.ltp.w_tilde
                   if isinstance(self.RSNN_EE_synapase.ltp.w_tilde, torch.Tensor)
                   else torch.tensor(self.RSNN_EE_synapase.ltp.w_tilde),
                   os.path.join(save_dir, f"RSNN_EE_w_tilde_{tag}.pt"))
        torch.save(self.inputs2RSNN_EE_synapase.ltp.w_tilde
                   if isinstance(self.inputs2RSNN_EE_synapase.ltp.w_tilde, torch.Tensor)
                   else torch.tensor(self.inputs2RSNN_EE_synapase.ltp.w_tilde),
                   os.path.join(save_dir, f"inputs2RSNN_EE_w_tilde_{tag}.pt"))

        # 3. Masks
        torch.save(self.RSNN_EE_synapase.RSNN_EE_mask.cpu(), os.path.join(save_dir, "RSNN_EE_mask.pt"))
        torch.save(self.RSNN_EI_synapase.RSNN_EI_mask.cpu(), os.path.join(save_dir, "RSNN_EI_mask.pt"))
        torch.save(self.RSNN_IE_synapase.IE_weights_mask.cpu(), os.path.join(save_dir, "RSNN_IE_mask.pt"))
        torch.save(self.RSNN_II_synapase.II_weights_mask.cpu(), os.path.join(save_dir, "RSNN_II_mask.pt"))
        torch.save(self.inputs2RSNN_EE_synapase.inputs_EE_mask.cpu(), os.path.join(save_dir, "inputs2RSNN_EE_mask.pt"))

        # 4. Neuron coordinates
        torch.save(self.RSNN_Exc_coordinates.cpu(), os.path.join(save_dir, "RSNN_Exc_coords.pt"))
        torch.save(self.RSNN_Inh_coordinates.cpu(), os.path.join(save_dir, "RSNN_Inh_coords.pt"))

        print(f"Full checkpoint saved to {save_dir} (tag={tag})")

    def show_mask_animation(self):
        self.inputs2RSNN_EE_synapase.show_animation()
        self.RSNN_EE_synapase.show_animation()
        self.RSNN_EI_synapase.show_animation()
        self.RSNN_IE_synapase.show_animation()
        self.RSNN_II_synapase.show_animation()

    def get_neurons_labels(self, loader, path, save_spike_inputs: bool = True):
        self.disable_plasticity()
        with torch.no_grad():
            if save_spike_inputs:
                N_exc = int(self.exc_neurons)
                N_samples = len(loader.dataset)
                T_use = int(self.T_training_on)

                feat_path  = f"{path}_spiketrains_uint8.dat"
                label_path = f"{path}_spiketrains_labels.npy"
                meta_path  = f"{path}_spiketrains_meta.json"

                # 以 memmap 方式建立大型陣列，避免暴記憶體
                X_mm = np.memmap(feat_path, dtype=np.uint8, mode="w+",
                                shape=(N_samples, N_exc, T_use))
                y_all = np.empty((N_samples,), dtype=np.int64)
                sidx = 0  # 目前寫到第幾個樣本
                
            if not os.path.exists(f"{path}.pt"):
                # put the input x into the model
                self.eval()

                self.neurons_labels = torch.zeros((self.exc_neurons, 10), device=self.device)
                # self.inputs_exc_spikes_trace = torch.zeros((28, 28), device=self.device)
                self.inputs_exc_spikes_trace = torch.zeros((self.inputs_size[2],self.inputs_size[3]), device=self.device)
                self.trace_fast = torch.zeros((self.inputs_size[2],self.inputs_size[3]), device=self.device)
                self.trace_slow = torch.zeros((self.inputs_size[2],self.inputs_size[3]), device=self.device)
                # self.each_pixels_labels = torch.zeros((self.exc_neurons, 28, 28), device=self.device)
                self.each_pixels_labels = torch.zeros((self.exc_neurons, self.inputs_size[2], self.inputs_size[3]), device=self.device)
                self.w_fast = 0.7
                self.w_slow = 0.3
                alpha_f = 1.0 - (self.dt / 5)
                alpha_s = 1.0 - (self.dt / 100)

                self.rf_accum = torch.zeros_like(self.each_pixels_labels)   # [N, 28, 28]
                self.post_spike_count = torch.zeros(self.exc_neurons, device=self.device)  # [N]
                self.trace_baseline_sum = torch.zeros((1,1,28,28), device=self.device)
                self.trace_baseline_count = 0

                print("Assigning classes to neurons")
                count = 0
                for x, y in tqdm(loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    B = x.shape[0]
                    for i in range(self.T_training_on):
                        # excitatory neurons 發出 spikes
                        exc_spikes, inh_spikes = self.forward(x) # spike = [batch size, neurons count]
                        # print((exc_spikes[0] > 0).nonzero(as_tuple=False).squeeze())
                        # print(exc_spikes[0] > 0)
                        fired = exc_spikes[0].bool() 
                        inputs_Exc_spikes = self.inputs_Exc_spikes[0]  # 取出 batch[0] 的 spikes → [neurons]
                        # d_inputs_Exc_spikes_trace = ((0 - self.inputs_exc_spikes_trace) + inputs_Exc_spikes)/20 # wrong time constant, use synapse time constant is good
                        # self.inputs_exc_spikes_trace = self.inputs_exc_spikes_trace + d_inputs_Exc_spikes_trace * self.dt
                        self.trace_fast = alpha_f * self.trace_fast + (self.dt / 5) * inputs_Exc_spikes
                        self.trace_slow = alpha_s * self.trace_slow + (self.dt / 100) * inputs_Exc_spikes
                        self.inputs_exc_spikes_trace = self.w_fast * self.trace_fast + self.w_slow * self.trace_slow
                        # print(inputs_Exc_spikes.shape) [1, 28, 28] [C, H, W]
                        # print("exc_spikes", exc_spikes)
                        # print("sum_of_exc_spikes", exc_spikes.sum()) 

                        if save_spike_inputs:
                            t_idx = i
                            # 將本步的 batch spikes 寫到 memmap
                            # 若 B>1，逐樣本寫入（你的 loader 幾乎是 B=1，這裡仍做通用）
                            sp_np = exc_spikes.to(torch.uint8).detach().cpu().numpy()  # [B,N_exc]
                            X_mm[sidx:sidx+B, :, t_idx] = sp_np

                        if i >= 300:
                            self.neurons_labels[:,y[0]] += exc_spikes[0]
                            # if exc neurons spikes, add the pixels label
                            # self.each_pixels_labels[exc_spikes[0].bool(), :] += inputs_Exc_spikes

                            self.rf_accum[fired] += self.inputs_exc_spikes_trace    # [28,28] 加到所有 fired neurons
                            self.post_spike_count[fired] += 1.0

                            # self.each_pixels_labels[exc_spikes[0].bool(), :] += self.inputs_exc_spikes_trace
                            # print("self.each_pixels_labels:", self.each_pixels_labels[2779])

                        self.trace_baseline_sum += self.inputs_exc_spikes_trace
                        self.trace_baseline_count += 1

                        self.step()
                    
                    zero_input = torch.zeros_like(x)
                    for t in range(self.T_training_off):
                        RSNN_Exc_spike, RSNN_Inh_spike = self.forward(zero_input)
                        fired = RSNN_Exc_spike[0].bool()
                        inputs_Exc_spikes = self.inputs_Exc_spikes[0]
                        self.trace_fast = alpha_f * self.trace_fast + (self.dt / 5) * inputs_Exc_spikes
                        self.trace_slow = alpha_s * self.trace_slow + (self.dt / 100) * inputs_Exc_spikes
                        self.inputs_exc_spikes_trace = self.w_fast * self.trace_fast + self.w_slow * self.trace_slow

                        self.neurons_labels[:, y[0]] += RSNN_Exc_spike[0]
                        self.rf_accum[fired] += self.inputs_exc_spikes_trace
                        self.post_spike_count[fired] += 1.0
                        self.trace_baseline_sum += self.inputs_exc_spikes_trace
                        self.trace_baseline_count += 1
                        self.step()
                    

                    eps = 1e-8
                    RF_mean = self.rf_accum / (self.post_spike_count.view(-1,1,1) + eps)  # E[trace | post=1]
                    trace_baseline = self.trace_baseline_sum / max(self.trace_baseline_count, 1)  # E[trace]
                    RF_sta = RF_mean - trace_baseline.squeeze(0).squeeze(0)  # baseline subtraction

                    self.each_pixels_labels = RF_sta.detach()  # 之後存檔/重建都用這個
                    
                    # spike-train 的標籤（每樣本一個）
                    if save_spike_inputs:
                        y_all[sidx:sidx+B] = y.view(-1).detach().cpu().numpy().astype(np.int64)
                        sidx += B

                print("Each Neurons Labels：",self.neurons_labels)
                print("Each Pixels Labels：",self.each_pixels_labels)
                # normalized_labels = F.normalize(self.neurons_labels, p=1, dim=1)
                torch.save(self.each_pixels_labels, f"{path}_each_pixels_labels.pt")
                torch.save(self.neurons_labels, f"{path}.pt")
                print("Neuron labels saved to", path)
                 # 存 spike-train 的 meta/標籤
                if save_spike_inputs:
                    X_mm.flush()
                    np.save(label_path, y_all)
                    meta = {
                        "N_samples": int(len(loader.dataset)),
                        "N_exc": int(self.exc_neurons),
                        "T_training_on": int(self.T_training_on),
                        "T_use": int(self.T_training_on),
                        "dtype": "uint8",
                        "feat_path": feat_path,
                        "label_path": label_path
                    }
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    print(f"[SAVED] spike-train features → {feat_path}")
                    print(f"[SAVED] labels → {label_path}")
                    print(f"[META ] {meta_path}")

            else:
                self.neurons_labels = torch.load(f"{path}.pt", map_location=self.device)
                print("Neuron labels loaded from", path)
    
    def eval_accuracy(self, loader, label_path=None):
        self.disable_plasticity()
        self.eval()
        correct, total = 0, 0

        if label_path != None:
            self.neurons_labels = torch.load(f"{label_path}.pt", map_location=self.device)

        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(self.device); y = y.to(self.device)
                # print(y)
                vote_scores = torch.zeros(10, device=self.device)
                for t in range(self.T_training_on):
                    exc_spikes, _ = self.forward(x)
                    if t >= 300:
                        vote_scores += torch.matmul(exc_spikes[0], self.neurons_labels)

                    self.step()

                zero_input = torch.zeros_like(x)
                for t in range(self.T_training_off):
                    exc_spikes, _ = self.forward(zero_input)
                    vote_scores += torch.matmul(exc_spikes[0], self.neurons_labels)
                    self.step()

                pred = vote_scores.argmax().item()
                correct += int(pred == y.item())
                total += 1

        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def test_all_methods(self, loader, top_k=200, class_templates=None, save_spike_inputs=True, path="./"):
        self.disable_plasticity()
        print("Start warm-up our Brain...")
        if save_spike_inputs:
            N_exc = int(self.exc_neurons)
            N_samples = len(loader.dataset)
            T_use = int(self.T_training_on)

            feat_path  = f"{path}_test_spiketrains_uint8.dat"
            label_path = f"{path}_test_spiketrains_labels.npy"
            meta_path  = f"{path}_test_spiketrains_meta.json"

            # 以 memmap 方式建立大型陣列，避免暴記憶體
            X_mm = np.memmap(feat_path, dtype=np.uint8, mode="w+",
                            shape=(N_samples, N_exc, T_use))
            y_all = np.empty((N_samples,), dtype=np.int64)
            sidx = 0  # 目前寫到第幾個樣本

        num_classes = 10
        confusion_default = torch.zeros((num_classes, num_classes), device=self.device)
        confusion_topk = torch.zeros((num_classes, num_classes), device=self.device)
        confusion_cosine = torch.zeros((num_classes, num_classes), device=self.device)
        confusion_decoder = torch.zeros((num_classes, num_classes), device=self.device) if class_templates is not None else None

        # 各類別總數
        total_per_class = torch.zeros(num_classes, device=self.device)

        self.eval()
        print("Warm-up Done. Start testing...")

        # Subtract row-mean so non-selective neurons contribute ~0 to voting
        row_mean = self.neurons_labels.mean(dim=1, keepdim=True)
        selective_labels = self.neurons_labels - row_mean

        # 記錄每種方法的正確率
        correct_default = 0
        correct_topk = 0
        correct_cosine = 0
        correct_decoder = 0
        total = 0

        max_fire_neurons = {}
        time_step_counter = torch.zeros(num_classes, self.T_training_on, self.exc_neurons,device = self.device)

        for x, y in tqdm(loader):
            x = x.to(self.device)
            y = y.to(self.device)
            B = x.size(0)
            vote_scores = torch.zeros(10, device=self.device)
            spike_record = torch.zeros(self.exc_neurons, device=self.device)

            for t in range(self.T_training_on):
                exc_spikes, _ = self.forward(x)
                if save_spike_inputs:
                    t_idx = t
                    # 將本步的 batch spikes 寫到 memmap
                    # 若 B>1，逐樣本寫入（你的 loader 幾乎是 B=1，這裡仍做通用）
                    sp_np = exc_spikes.to(torch.uint8).detach().cpu().numpy()  # [B,N_exc]
                    X_mm[sidx:sidx+B, :, t_idx] = sp_np

                if t >= 300:
                    spikes = exc_spikes[0]
                    vote_scores += torch.matmul(spikes, selective_labels)
                    spike_record += spikes

                # print(time_step_counter[:,t, y].shape)
                # print(exc_spikes.shape)
                time_step_counter[y, t, :] += exc_spikes
                self.step()

            # 執行 testing-off: zero input, count all spikes into votes
            zero_input = torch.zeros_like(x)
            for t in range(self.T_training_off):
                exc_spikes, _ = self.forward(zero_input)
                spikes = exc_spikes[0]
                vote_scores += torch.matmul(spikes, selective_labels)
                spike_record += spikes
                self.step()

            # spike-train 的標籤（每樣本一個）
            if save_spike_inputs:
                y_all[sidx:sidx+B] = y.view(-1).detach().cpu().numpy().astype(np.int64)
                sidx += B
            
            pred_default = vote_scores.argmax().item()
            confusion_default[y.item(), pred_default] += 1
            if pred_default == y.item():
                correct_default += 1
                # correct_default_cls[y.item()] += 1

            topk_idx = torch.topk(spike_record, top_k).indices
            topk_votes = selective_labels[topk_idx]
            pred_topk = topk_votes.sum(dim=0).argmax().item()
            confusion_topk[y.item(), pred_topk] += 1
            if pred_topk == y.item():
                correct_topk += 1
                # correct_topk_cls[y.item()] += 1

            # 方法三：Cosine similarity (using selective_labels)
            spike_norm = F.normalize(spike_record, p=2, dim=0).unsqueeze(0)
            label_norm = F.normalize(selective_labels.T, p=2, dim=0)
            cos_sim = torch.matmul(spike_norm, label_norm.T).squeeze()
            pred_cosine = cos_sim.argmax().item()
            confusion_cosine[y.item(), pred_cosine] += 1
            if pred_cosine == y.item():
                correct_cosine += 1
                # correct_cosine_cls[y.item()] += 1

            # 方法四：Decoder reconstruction compare to class_templates
            if class_templates is not None:
                recon_map = (spike_record.view(-1, 1, 1) * self.each_pixels_labels).sum(dim=0)
                dists = [F.mse_loss(recon_map, class_templates[i]) for i in range(num_classes)]
                pred_decoder = torch.argmin(torch.tensor(dists)).item()
                confusion_decoder[y.item(), pred_decoder] += 1
                if pred_decoder == y.item():
                    correct_decoder += 1
                    # correct_decoder_cls[y.item()] += 1

            total += 1
        
        max_neuron_idx = torch.argmax(time_step_counter, dim=2)

        # 轉成你要的 dict 形式：{class_i: [t=0 的 neuron, t=1 的 neuron, ...]}
        max_fire_neurons = {int(c): max_neuron_idx[c].tolist()
                            for c in range(num_classes)}
        # print(max_fire_neurons)
        save_path = "./max_fire_neurons.json"

        with open(save_path, "w") as f:
            json.dump(max_fire_neurons, f)

        print(f"已儲存至 {save_path}")
        
        if save_spike_inputs:
            X_mm.flush()
            np.save(label_path, y_all)
            meta = {
                "N_samples": int(len(loader.dataset)),
                "N_exc": int(self.exc_neurons),
                "T_training_on": int(self.T_training_on),
                "T_use": int(self.T_training_on),
                "dtype": "uint8",
                "feat_path": feat_path,
                "label_path": label_path
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[SAVED] spike-train features → {feat_path}")
            print(f"[SAVED] labels → {label_path}")
            print(f"[META ] {meta_path}")
            
        print(f"[Default Argmax]      Accuracy: {correct_default / total * 100:.2f}%")
        print(f"[Top-{top_k} Neurons] Accuracy: {correct_topk / total * 100:.2f}%")
        print(f"[Cosine Similarity]   Accuracy: {correct_cosine / total * 100:.2f}%")
        if class_templates is not None:
            print(f"[Decoder Matching]    Accuracy: {correct_decoder / total * 100:.2f}%")
        
        # acc_default_per_class = (correct_default_cls / total_per_class).cpu().numpy() * 100
        # acc_topk_per_class = (correct_topk_cls / total_per_class).cpu().numpy() * 100
        # acc_cosine_per_class = (correct_cosine_cls / total_per_class).cpu().numpy() * 100

        # if class_templates is not None:
        #     acc_decoder_per_class = (correct_decoder_cls / total_per_class).cpu().numpy() * 100
        # else:
        #     acc_decoder_per_class = None
        
        import matplotlib.pyplot as plt
        import seaborn as sns

        def plot_confusion(cm, title):
            cm_np = cm.cpu().numpy()
            plt.figure(figsize=(8,6))
            sns.heatmap(cm_np, annot=True, fmt=".0f", cmap="YlGnBu", 
                        xticklabels=[str(i) for i in range(num_classes)],
                        yticklabels=[str(i) for i in range(num_classes)])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(title)
            # save
            plt.savefig(f"./{title}.png", dpi=200, bbox_inches='tight')
            plt.show()

        # 畫所有方法的 confusion matrix
        plot_confusion(confusion_default, "Confusion Matrix - Default Argmax")
        plot_confusion(confusion_topk, f"Confusion Matrix - Top-{top_k} Neurons")
        plot_confusion(confusion_cosine, "Confusion Matrix - Cosine Similarity")
        if confusion_decoder is not None:
            plot_confusion(confusion_decoder, "Confusion Matrix - Decoder Matching")
        

    # ========= Receptive field (weights-based) =========
    def get_rf_weights(self, neuron_indices=None, normalize='minmax'):
        """
        傳回 [K, 28, 28] 的 RF tensor（K 為選取的神經元數）
        normalize: 'minmax' | 'zscore' | None   （逐顆 neuron 做）
        """
        # 784 x N_exc
        W = self.inputs2RSNN_EE_synapase.inputs_EE_weights.detach().clone()  # [784, N]
        if neuron_indices is None:
            neuron_indices = torch.arange(W.shape[1], device=W.device)
        else:
            neuron_indices = torch.as_tensor(neuron_indices, device=W.device)

        # 取出欄位 → [784, K] → [K, 28, 28]
        # maps = W[:, neuron_indices].T.reshape(-1, 28, 28)
        maps = W[:, neuron_indices].T.reshape(-1, self.inputs_size[2], self.inputs_size[3])

        if normalize == 'minmax':
            vmin = maps.view(maps.size(0), -1).min(dim=1)[0][:, None, None]
            vmax = maps.view(maps.size(0), -1).max(dim=1)[0][:, None, None]
            maps = (maps - vmin) / (vmax - vmin + 1e-8)
        elif normalize == 'zscore':
            m = maps.view(maps.size(0), -1).mean(dim=1)[:, None, None]
            s = maps.view(maps.size(0), -1).std(dim=1)[:, None, None]
            maps = (maps - m) / (s + 1e-8)

        return maps  # device 與權重相同

    def plot_rf_grid(self, neuron_indices, cols=8, normalize='minmax',
                     cmap='Blues', save_path=None, suptitle=None):
        """
        把多顆神經元的 RF 以網格顯示。
        """
        import matplotlib.pyplot as plt
        rfs = self.get_rf_weights(neuron_indices, normalize=normalize).cpu()
        K = rfs.shape[0]
        rows = math.ceil(K / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
        axes = axes.ravel() if K > 1 else [axes]
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis('off')
            if i < K:
                im = ax.imshow(rfs[i], cmap=cmap, vmin=rfs[i].min(), vmax=rfs[i].max())
                ax.set_title(f"n={neuron_indices[i]}", fontsize=8)
        # 右側加同一個 colorbar（與論文圖示相近）
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cax, label=r"$w_{ij}^{ext}$ (a.u.)")
        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()

    def plot_topk_rf_per_class(self, cls, k=4, cols=None, normalize='minmax',
                               cmap='Blues', save_path=None):
        """
        先用你在 get_neurons_labels() 算好的 neurons_labels，
        選該類別貢獻度最高的前 k 顆 excitatory neurons，畫 RF。
        回傳這些 neuron 的 index，方便你記錄。
        """
        assert hasattr(self, "neurons_labels"), \
            "請先呼叫 get_neurons_labels(…) 以建立 self.neurons_labels"

        # 取 top-k indices
        topk = torch.topk(self.neurons_labels[:, cls], k=k).indices.detach().cpu().tolist()
        if cols is None: cols = k
        self.plot_rf_grid(topk, cols=cols, normalize=normalize, cmap=cmap,
                          save_path=save_path,
                          suptitle=f"Class {cls} — top-{k} neurons RF")
        return topk


    #########################
    # get spikes funtion    #
    #########################
    def get_inputs_Exc_spikes(self):
        return self.inputs_Exc_spikes
    def get_inputs_Inh_spikes(self):
        return self.inputs_Inh_spikes
