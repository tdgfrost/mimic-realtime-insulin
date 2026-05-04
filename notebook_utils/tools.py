from IPython import display
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Optional, Dict, Union, Iterable
import torch


@dataclass
class Batch:
    states: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None
    reward_markers: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None
    next_actions: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    infos: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]] = None


class DataLoader:
    """
    Zero-overhead dataloader for fully GPU-resident datasets.
    Relies on PyTorch's native asynchronous CUDA execution.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = dataset.device
        self.n = len(dataset)

    def __iter__(self):
        # Generate all indices for the epoch at once on the GPU
        if self.shuffle:
            indices = torch.randperm(self.n, device=self.device)
        else:
            indices = torch.arange(self.n, device=self.device)

        # Slice directly in the main thread; PyTorch handles the async dispatch
        for i in range(0, self.n, self.batch_size):
            idx = indices[i : i + self.batch_size]
            raw_batch = self.dataset[idx]
            yield Batch(**raw_batch)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def update_plots(current_idx, iters, losses, aurocs=None, v_s0=None):
    display.clear_output(wait=True)
    n_plots = 1
    if aurocs is not None:
        n_plots += 1
    if v_s0 is not None:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]
    ax_loss = axes[0]
    plot_idx = 1

    if isinstance(losses, dict):
        for loss_name, loss_values in losses.items():
            linestyle = '-' if 'train' in loss_name.lower() else '--'
            ax_loss.plot(iters, loss_values, label=loss_name, linewidth=1.5, linestyle=linestyle)
        ax_loss.set_ylabel("Loss Magnitude")
    else:
        ax_loss.plot(iters, losses, label='Loss', color='tab:blue', linewidth=1.5)
        ax_loss.set_ylabel("Cross Entropy")

    # Set logarithmic scale for the loss plot
    ax_loss.set_yscale('log')
    ax_loss.set_title(f"Model Losses")
    ax_loss.set_xlabel("Epoch")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    if aurocs is not None:
        ax_auroc = axes[plot_idx]
        ax_mae = ax_auroc.twinx()  # Create the secondary y-axis
        plot_idx += 1

        lines = []
        labels = []

        for metric_name, scores in aurocs.items():
            linestyle = '-' if 'train' in metric_name.lower() else '--'

            # Route MAE to the secondary axis
            if 'MAE' in metric_name:
                line = ax_mae.plot(iters, scores, label=metric_name, linestyle=linestyle, linewidth=2)
                ax_mae.set_ylabel("Mean Absolute Error (MAE)")
            else:
                line = ax_auroc.plot(iters, scores, label=metric_name, linestyle=linestyle)

            # Collect handles for a unified legend
            lines.extend(line)
            labels.append(metric_name)

        ax_auroc.set_title("Validation Metrics")
        ax_auroc.set_xlabel("Epoch")
        ax_auroc.set_ylabel("AUROC Score")
        ax_auroc.set_ylim(0.5, 1.0)
        ax_mae.set_ylim(0.0, 1.5)
        ax_auroc.grid(True, alpha=0.3)

        # Combine legends from both axes
        # We attach the legend to ax_mae because it is the top-most layer
        leg = ax_mae.legend(lines, labels, loc='upper left')

        # Set zorder to a high value to force it to the front
        leg.set_zorder(100)

        # Ensure the legend background is opaque so lines don't show through
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_facecolor('white')

    if v_s0 is not None:
        ax_v = axes[plot_idx]
        if isinstance(v_s0, dict):
            for v_name, v_values in v_s0.items():
                linestyle = '-' if 'train' in v_name.lower() else '--'
                ax_v.plot(iters, v_values, label=v_name, linewidth=1.5, linestyle=linestyle)
        else:
            ax_v.plot(iters, v_s0, label='$V(S_0)$', color='tab:green', linewidth=1.5)

        # Rename the title as requested
        ax_v.set_title("Predicted $V(S_0)$")
        ax_v.set_xlabel("Epoch")
        ax_v.set_ylabel("Predicted Value")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend()

    plt.tight_layout()
    plt.show()