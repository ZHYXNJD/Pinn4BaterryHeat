import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

class Visualizer:
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.loss_history: Dict[str, List[float]] = {
            "total": [],
            "data":[],
            "residual": [],
            "boundary": [],
            "initial": [],
            "fluid_residual": [],
            "fluid_inlet": [],
            "interface": [],
        }
        self._setup_style()

    def _setup_style(self):
        try:
            plt.style.use('seaborn-v0_8-paper')
        except OSError:
            try:
                plt.style.use('seaborn-paper')
            except OSError:
                pass 
        
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "text.usetex": False, 
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
        })

    def log_loss(
        self,
        total: float,
        data: float,
        res: float,
        bc: float,
        ic: float,
        fluid_res: float = 0.0,
        fluid_inlet: float = 0.0,
        interface: float = 0.0,
    ):
        self.loss_history["total"].append(total)
        self.loss_history["data"].append(data)
        self.loss_history["residual"].append(res)
        self.loss_history["boundary"].append(bc)
        self.loss_history["initial"].append(ic)
        self.loss_history["fluid_residual"].append(fluid_res)
        self.loss_history["fluid_inlet"].append(fluid_inlet)
        self.loss_history["interface"].append(interface)

    def plot_loss(self, epoch: int):
        fig, ax = plt.subplots(figsize=(3.5, 2.5)) 
        
        epochs = range(1, len(self.loss_history["total"]) + 1)
        
        ax.plot(epochs, self.loss_history["total"], label="Total", color="#d62728")
        ax.plot(epochs, self.loss_history["data"], label="Data")
        ax.plot(epochs, self.loss_history["residual"], label="Residual", color="#1f77b4", linestyle="--")
        ax.plot(epochs, self.loss_history["boundary"], label="Boundary", color="#2ca02c", linestyle=":")
        ax.plot(epochs, self.loss_history["initial"], label="Initial", color="#ff7f0e", linestyle="-.")
        if any(self.loss_history["fluid_residual"]):
            ax.plot(
                epochs,
                self.loss_history["fluid_residual"],
                label="Fluid residual",
                color="#9467bd",
                linestyle="--",
            )
        if any(self.loss_history["fluid_inlet"]):
            ax.plot(
                epochs,
                self.loss_history["fluid_inlet"],
                label="Fluid inlet",
                color="#8c564b",
                linestyle=":",
            )
        if any(self.loss_history["interface"]):
            ax.plot(
                epochs,
                self.loss_history["interface"],
                label="Interface",
                color="#17becf",
                linestyle="-.",
            )
        
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training Loss Convergence")
        ax.legend(frameon=False)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "loss_history.png")
        plt.close(fig)
        print(f"Saved loss plot to {self.save_dir / 'loss_history.png'}")

    def plot_temperature(
        self,
        model,
        geometry,
        device,
        epoch: int,
        condition,
        t_max_val: float,
        init_temp: float,
        name_suffix: str = "",
        normalize_fn=None,
    ):
        half_x = geometry.length_x / 2.0
        half_y = geometry.length_y / 2.0
        z_plane = 0.0
        x = np.linspace(-half_x, half_x, 100)
        y = np.linspace(-half_y, half_y, 100)
        X, Y = np.meshgrid(x, y)

        # Align visualization sampling with the centered training coordinates.
        
        N = X.size
        xyz = np.zeros((N, 3), dtype=np.float32)
        xyz[:, 0] = X.flatten()
        xyz[:, 1] = Y.flatten()
        xyz[:, 2] = z_plane
        
        t_max = t_max_val
        t = np.full((N, 1), t_max, dtype=np.float32)
        
        xyz_tensor = torch.tensor(xyz, device=device)
        t_tensor = torch.tensor(t, device=device)
        
        cond = condition
        cond_vec = cond.vector().to(device).unsqueeze(0).repeat(N, 1)
        
        if normalize_fn is None:
            xyz_n = geometry.normalized(xyz_tensor)
            t_n = (t_tensor / t_max) * 2.0 - 1.0
            norm_coords = torch.cat([xyz_n, t_n], dim=1)
        else:
            norm_coords = normalize_fn(xyz_tensor, t_tensor)
        
        model.eval()
        with torch.no_grad():
            T_net = model(norm_coords, cond_vec)
            T = T_net + init_temp
        
        T_pred = T.cpu().numpy().reshape(X.shape)
        
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        
        im = ax.contourf(X * 1000, Y * 1000, T_pred, levels=50, cmap="inferno")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Temperature (K)")
        
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"T (z={z_plane*1000:.1f}mm, t={t_max}s)\n{cond.name} {name_suffix}")
        
        filename = f"temperature_{cond.name}_{name_suffix}.png" if name_suffix else f"temperature_{cond.name}_epoch_{epoch:06d}.png"
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close(fig)
        print(f"Saved temperature plot to {self.save_dir / filename}")
