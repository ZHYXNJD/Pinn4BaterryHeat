import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple

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
        # 按训练步长记录的 loss，用于按 step 绘制
        self.step_loss_history: Dict[str, List[float]] = {
            "total": [],
            "data": [],
            "residual": [],
            "boundary": [],
            "initial": [],
            "fluid_residual": [],
            "fluid_inlet": [],
            "interface": [],
        }
        self.step_indices: List[int] = []
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

    def log_loss_step(
        self,
        step: int,
        total: float,
        data: float,
        res: float,
        bc: float,
        ic: float,
        fluid_res: float = 0.0,
        fluid_inlet: float = 0.0,
        interface: float = 0.0,
    ):
        """按训练步长记录 loss，用于按 step 绘制曲线。"""
        self.step_indices.append(step)
        self.step_loss_history["total"].append(total)
        self.step_loss_history["data"].append(data)
        self.step_loss_history["residual"].append(res)
        self.step_loss_history["boundary"].append(bc)
        self.step_loss_history["initial"].append(ic)
        self.step_loss_history["fluid_residual"].append(fluid_res)
        self.step_loss_history["fluid_inlet"].append(fluid_inlet)
        self.step_loss_history["interface"].append(interface)

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

    def plot_loss_by_step(self):
        """按训练步长（step）绘制 loss 曲线，适用于每 epoch 步数较多时的细粒度查看。"""
        if not self.step_indices:
            return
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        steps = self.step_indices
        ax.plot(steps, self.step_loss_history["total"], label="Total", color="#d62728")
        ax.plot(steps, self.step_loss_history["data"], label="Data")
        ax.plot(steps, self.step_loss_history["residual"], label="Residual", color="#1f77b4", linestyle="--")
        ax.plot(steps, self.step_loss_history["boundary"], label="Boundary", color="#2ca02c", linestyle=":")
        ax.plot(steps, self.step_loss_history["initial"], label="Initial", color="#ff7f0e", linestyle="-.")
        if any(self.step_loss_history["fluid_residual"]):
            ax.plot(
                steps,
                self.step_loss_history["fluid_residual"],
                label="Fluid residual",
                color="#9467bd",
                linestyle="--",
            )
        if any(self.step_loss_history["fluid_inlet"]):
            ax.plot(
                steps,
                self.step_loss_history["fluid_inlet"],
                label="Fluid inlet",
                color="#8c564b",
                linestyle=":",
            )
        if any(self.step_loss_history["interface"]):
            ax.plot(
                steps,
                self.step_loss_history["interface"],
                label="Interface",
                color="#17becf",
                linestyle="-.",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training Loss Convergence (by Step)")
        ax.legend(frameon=False)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.save_dir / "loss_history_by_step.png")
        plt.close(fig)
        print(f"Saved loss-by-step plot to {self.save_dir / 'loss_history_by_step.png'}")

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

    def plot_temperature_yz_slice(
        self,
        model,
        geometry,
        device,
        condition,
        t_max_val: float,
        init_temp: float,
        temperature_max,
        temperature_min,
        x_positions_mm: List[float] = None,
        time_points: List[float] = None,
        normalize_fn=None,
        save_dir=None,
        config_name: str = "",
        model_to_temp_fn=None,
    ):
        """Plot yz-plane temperature heatmaps at fixed x positions for multiple time points.

        Args:
            x_positions_mm: list of x positions in mm for cross-sections, e.g. [33, 132, 264, 396]
            time_points: list of times in seconds, e.g. [300, 600, 900, 1200, 1500, 1800]
        """
        if time_points is None:
            time_points = [t_max_val]
        if x_positions_mm is None:
            x_positions_mm = [33.0]
        x_list = [float(x) for x in x_positions_mm]

        out_dir = Path(save_dir) if save_dir is not None else self.save_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        half_y = geometry.length_y / 2.0
        half_z = geometry.length_z / 2.0


        y = np.linspace(-half_y, half_y, 100)
        z = np.linspace(-half_z, half_z, 100)
        Y, Z = np.meshgrid(y, z)

        N = Y.size
        cond_vec = condition.vector().to(device).unsqueeze(0).repeat(N, 1)

        model.eval()
        with torch.no_grad():
            for x_mm in x_list:
                x_m = (x_mm / 1000.0) - (geometry.length_x / 2.0)

                xyz = np.zeros((N, 3), dtype=np.float32)
                xyz[:, 0] = x_m
                xyz[:, 1] = Y.flatten()
                xyz[:, 2] = Z.flatten()
                xyz_tensor = torch.tensor(xyz, device=device)

                for t_val in time_points:
                    t_val = min(float(t_val), t_max_val)
                    t = np.full((N, 1), t_val, dtype=np.float32)
                    t_tensor = torch.tensor(t, device=device)

                    if normalize_fn is None:
                        xyz_n = geometry.normalized(xyz_tensor)
                        t_n = (t_tensor / t_max_val) * 2.0 - 1.0
                        norm_coords = torch.cat([xyz_n, t_n], dim=1)
                    else:
                        norm_coords = normalize_fn(xyz_tensor, t_tensor)

                    T_net = model(norm_coords, cond_vec)
                    T = model_to_temp_fn(T_net) if model_to_temp_fn is not None else T_net + init_temp
                    T_pred = T.cpu().numpy().reshape(Y.shape)

                    fig, ax = plt.subplots(figsize=(3.5, 3.0))
                    im = ax.contourf(Y * 1000, Z * 1000, T_pred, levels=50, cmap="inferno",vmax=temperature_max+5, vmin=temperature_min-5)
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("Temperature (K)")
                    ax.set_xlabel("y (mm)")
                    ax.set_ylabel("z (mm)")
                    ax.set_title(f"T (x={x_mm:.0f}mm, t={t_val:.0f}s)\n{condition.name}")
                    cfg_suffix = f"_{config_name}" if config_name else ""
                    filename = f"heatmap_yz_x{x_mm:.0f}mm_{condition.name}{cfg_suffix}_t{int(t_val)}s.png"
                    plt.tight_layout()
                    plt.savefig(out_dir / filename)
                    plt.close(fig)
                    print(f"Saved heatmap to {out_dir / filename}")

    def plot_prediction_vs_true(
        self,
        df: pd.DataFrame,
        point_coords: List[Tuple[float, float, float]],
        condition_name: str,
        save_path: str,
        config_name: str = "",
    ):
        """Plot predicted vs true temperature over time for selected spatial points.

        Args:
            df: DataFrame with columns x_mm, y_mm, z_mm, t, temperature_true, temperature_pred
            point_coords: List of 3 (x, y, z) tuples in mm
            condition_name: Condition name for title
            save_path: Path to save the figure
            config_name: Optional config name suffix
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        colors = ["#1f77b4", "#2ca02c", "#d62728"]

        for i, (x_mm, y_mm, z_mm) in enumerate(point_coords):
            mask = (
                (np.isclose(df["x_mm"], x_mm))
                & (np.isclose(df["y_mm"], y_mm))
                & (np.isclose(df["z_mm"], z_mm))
            )
            sub = df.loc[mask].sort_values("t")
            if sub.empty:
                continue
            label = f"({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f}) mm"
            c = colors[i % len(colors)]
            ax.plot(
                sub["t"],
                sub["temperature_true"],
                "-",
                color=c,
                linewidth=1.5,
                label=f"{label} true",
            )
            ax.plot(
                sub["t"],
                sub["temperature_pred"],
                "--",
                color=c,
                linewidth=1.0,
                label=f"{label} pred",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(f"Predicted vs True: {condition_name}")
        ax.legend(frameon=False, ncol=2, fontsize=6)
        ax.grid(True, ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved pred vs true plot to {save_path}")
