from IPython import display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Sequence, Union, Iterable
import glob
import os
import random
import numpy as np
import torch
import re
from scipy import stats


@dataclass
class Batch:
    states: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None
    reward_markers: Optional[torch.Tensor] = None
    next_states: Optional[torch.Tensor] = None
    next_actions: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    infos: Optional[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]] = None
    # Row indices of this batch within the dataset, for looking up anything that
    # has been precomputed for the whole split (e.g. frozen policy actions).
    idx: Optional[torch.Tensor] = None


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


def update_plots(current_idx, iters, losses, aurocs=None, v_s0=None, title=None):
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
    ax_loss.legend(loc='upper left')

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
                linestyle = '--' if 'train' in v_name.lower() else '-'

                # Check if v_values is a scalar and plot a horizontal line if true
                if isinstance(v_values, (int, float)):
                    ax_v.axhline(y=v_values, label=v_name, linewidth=1.5, linestyle=linestyle)
                else:
                    ax_v.plot(iters, v_values, label=v_name, linewidth=1.5, linestyle=linestyle)
        else:
            # Also handle the case where the naked v_s0 is passed as a scalar
            if isinstance(v_s0, (int, float)):
                ax_v.axhline(y=v_s0, label='$V(S_0)$', color='tab:green', linewidth=1.5)
            else:
                ax_v.plot(iters, v_s0, label='$V(S_0)$', color='tab:green', linewidth=1.5)

        # Rename the title as requested
        ax_v.set_title("Predicted $V(S_0)$")
        ax_v.set_xlabel("Epoch")
        ax_v.set_ylabel("Predicted Value")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc='upper left')

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# =============================================================================
#   PLOTTING UMAP
# =============================================================================
def plot_latent_space(
    latent_2d, 
    all_bms, 
    all_stops, 
    all_delta_changes, 
    all_insulin_rates, 
    all_3d_alive, 
    all_episodes, 
    all_hours_remaining,
    title_size=14, 
    cbar_label_size=13, 
    tick_size=11, 
    legend_size=12,
    save_path=None, # "./latent_space_figure_2x3.pdf"
):
    
    # 3. Generate the Figure (Updated to 2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    
    # --- ROW 1 ---
    
    # Panel A: Current BM
    norm_bm = mcolors.LogNorm(vmin=np.min(all_bms), vmax=np.max(all_bms))
    
    sc_bm = axes[0, 0].scatter(
        latent_2d[:, 0], latent_2d[:, 1], 
        c=all_bms, cmap='magma', norm=norm_bm, 
        s=5, alpha=0.6
    )
    axes[0, 0].set_title("Latent Space by Current Blood Glucose (Log Scale)", fontsize=title_size)
    axes[0, 0].axis('off')
    
    # Force uniform axis sizing
    div_A = make_axes_locatable(axes[0, 0])
    cax_A = div_A.append_axes("right", size="5%", pad=0.1)
    
    cb_bm = fig.colorbar(sc_bm, cax=cax_A)
    cb_bm.set_label("Blood Glucose (mmol/L)", fontsize=cbar_label_size)
    cb_bm.ax.tick_params(labelsize=tick_size)
    
    bm_ticks = [2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28]
    cb_bm.set_ticks(bm_ticks)
    cb_bm.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    cb_bm.ax.yaxis.set_minor_formatter(ticker.NullFormatter()) 
    
    # Panel B: Actions
    stop_idx = all_stops == 1.0
    cont_idx = ~stop_idx
    
    norm = mcolors.TwoSlopeNorm(vmin=-5.5, vcenter=0.0, vmax=5.5)
    
    sc_action = axes[0, 1].scatter(
        latent_2d[cont_idx, 0], latent_2d[cont_idx, 1],
        c=all_delta_changes[cont_idx], cmap='RdBu_r', norm=norm,
        s=5, alpha=0.7
    )
    
    axes[0, 1].scatter(
        latent_2d[stop_idx, 0], latent_2d[stop_idx, 1],
        color='#9d4edd', s=5, alpha=0.9, label="Insulin Stopped"
    )
    
    axes[0, 1].set_title("Latent Space by Insulin Action", fontsize=title_size)
    axes[0, 1].axis('off')
    
    # Force uniform axis sizing
    div_B = make_axes_locatable(axes[0, 1])
    cax_B = div_B.append_axes("right", size="5%", pad=0.1)
    
    cbar = fig.colorbar(sc_action, cax=cax_B)
    cbar.set_label("Delta Insulin Rate (U/hr)", fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    
    axes[0, 1].legend(loc='upper left', markerscale=3, fontsize=legend_size)
    
    # Panel C: Current Insulin Rate (Absolute)
    ins_vmax = np.percentile(all_insulin_rates, 99.9)
    boundaries = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35]
    norm_ins = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=280)
    
    sc_ins = axes[0, 2].scatter(
        latent_2d[:, 0], latent_2d[:, 1], 
        c=all_insulin_rates, cmap='plasma', norm=norm_ins, 
        s=5, alpha=0.6
    )
    axes[0, 2].set_title("Latent Space by Current Insulin Rate", fontsize=title_size)
    axes[0, 2].axis('off')
    
    # Force uniform axis sizing
    div_C = make_axes_locatable(axes[0, 2])
    cax_C = div_C.append_axes("right", size="5%", pad=0.1)
    
    cb_ins = fig.colorbar(sc_ins, cax=cax_C)
    cb_ins.set_label("Insulin Rate (U/hr)", fontsize=cbar_label_size)
    cb_ins.ax.tick_params(labelsize=tick_size)
    cb_ins.set_ticks(boundaries)
    cb_ins.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    
    # --- ROW 2 ---
    
    # Panel D: 3-Day Mortality
    alive_idx = all_3d_alive == 1.0
    dead_idx = all_3d_alive == 0.0
    
    axes[1, 0].scatter(latent_2d[alive_idx, 0], latent_2d[alive_idx, 1], color='lightgray', s=5, alpha=0.3, label="Alive (3-Day)")
    axes[1, 0].scatter(latent_2d[dead_idx, 0], latent_2d[dead_idx, 1], color='crimson', s=10, alpha=0.8, label="Deceased (3-Day)")
    axes[1, 0].set_title("Latent Space by 3-Day Mortality", fontsize=title_size)
    axes[1, 0].axis('off')
    axes[1, 0].legend(loc='lower left', markerscale=3, fontsize=legend_size)
    
    # Create empty space to align perfectly with other panels
    div_D = make_axes_locatable(axes[1, 0])
    cax_D = div_D.append_axes("right", size="5%", pad=0.1)
    cax_D.axis('off')
    
    # Panel E: Trajectories
    axes[1, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], c='lightgray', s=5, alpha=0.15, zorder=1)
    
    unique_episodes = np.unique(all_episodes)[1:3]
    trajectory_cmaps = [mpl.colormaps['Greens'], mpl.colormaps['PuBu']]
    
    for i, ep in enumerate(unique_episodes):
        idx = np.where(np.array(all_episodes) == ep)[0]
    
        ep_x = latent_2d[idx, 0]
        ep_y = latent_2d[idx, 1]
        n_steps = len(ep_x)
    
        cmap = trajectory_cmaps[i % len(trajectory_cmaps)]
    
        for step in range(n_steps - 1):
            progress = (step + 1) / n_steps
            color = cmap(0.4 + 0.6 * progress)
    
            axes[1, 1].annotate(
                "",
                xy=(ep_x[step+1], ep_y[step+1]),
                xytext=(ep_x[step], ep_y[step]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2,
                    mutation_scale=15,
                    alpha=0.8
                ), zorder=3
            )
    
            axes[1, 1].scatter(ep_x[step], ep_y[step], color=color, s=20, zorder=4)
    
        start_label = "Start" if i == 0 else None
        end_label = "End" if i == 0 else None
    
        axes[1, 1].scatter(ep_x[0], ep_y[0], marker='*', color=cmap(0.4), edgecolor='black',
                        s=150, zorder=5, label=start_label)
    
        axes[1, 1].scatter(ep_x[-1], ep_y[-1], marker='X', color=cmap(1.0), edgecolor='black',
                        s=100, zorder=5, label=end_label)
    
    axes[1, 1].set_title("Temporal Trajectories", fontsize=title_size)
    axes[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_size)
    axes[1, 1].axis('off')
    
    # Create empty space to align perfectly with other panels
    div_E = make_axes_locatable(axes[1, 1])
    cax_E = div_E.append_axes("right", size="5%", pad=0.1)
    cax_E.axis('off') 
    
    # Panel F: Hours Remaining
    state_idx = all_hours_remaining > 0
    norm_time_remaining = mcolors.LogNorm(vmin=np.min(all_hours_remaining[state_idx]), vmax=np.max(all_hours_remaining[state_idx]))
    sc_time = axes[1, 2].scatter(
        latent_2d[state_idx, 0], latent_2d[state_idx, 1], 
        c=all_hours_remaining[state_idx], cmap='viridis_r', norm=norm_time_remaining,
        s=5, alpha=0.6
    )
    axes[1, 2].set_title("Latent Space by Hours Remaining (Log Scale)", fontsize=title_size)
    axes[1, 2].axis('off')
    
    # Force uniform axis sizing
    div_F = make_axes_locatable(axes[1, 2])
    cax_F = div_F.append_axes("right", size="5%", pad=0.1)
    
    cb_time = fig.colorbar(sc_time, cax=cax_F)
    cb_time.set_label("Hours Remaining in Episode", fontsize=cbar_label_size)
    cb_time.ax.tick_params(labelsize=tick_size)
    cb_time.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

# =============================================================================
#   MULTI-SEED HELPERS
# =============================================================================
def set_seed(seed: Optional[int]):
    """
    Seed every RNG that affects training. A seed of `None` is a no-op, which
    keeps the single-run (default) workflow byte-for-byte as it was before.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_tag(**seeds: Optional[int]) -> str:
    """
    Build a filename suffix from the seeds identifying a run, skipping any that
    are `None`. `run_tag(seed=None)` -> '' (i.e. the original filenames are
    preserved when multi-seed mode is off), `run_tag(p=0, f=3)` -> '_p0_f3'.
    """
    parts = [f'{name}{value}' for name, value in seeds.items() if value is not None]
    return ('_' + '_'.join(parts)) if parts else ''


# =============================================================================
#   CRASH-SAFE CHECKPOINTING
# =============================================================================
def _atomic_torch_save(obj: Any, path: str):
    """Write via a temporary file so an interrupted save cannot corrupt `path`."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f'{path}.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _get_rng_state() -> Dict[str, Any]:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: Optional[Dict[str, Any]]):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


class Checkpoint:
    """
    Epoch-level checkpointing, so that an interrupted training run resumes from
    the last completed epoch instead of starting again from scratch.

    `history` is a dict of the *live* Python lists the training loop appends its
    metrics to. They are saved after every epoch and refilled in place on
    resume, so the body of the training loop does not need to change.

    Usage:
        ckpt = Checkpoint(path, models={'policy': model},
                          optimizers={'policy': optimizer},
                          history={'train_loss_history': train_loss_history})
        for epoch in range(ckpt.resume(), EPOCHS):
            ...
            ckpt.save(epoch + 1)
        ckpt.finalise('./saved_models/model.pt', model.state_dict())
    """

    def __init__(
            self,
            path: str,
            models: Optional[Dict[str, torch.nn.Module]] = None,
            optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
            history: Optional[Dict[str, list]] = None,
            enabled: bool = True,
    ):
        self.path = path
        self.models = models or {}
        self.optimizers = optimizers or {}
        self.history = history or {}
        self.enabled = enabled
        self.next_epoch = 0
        self.completed = False

    def resume(self) -> int:
        """Restore any saved state and return the epoch index to start from."""
        if not self.enabled or not os.path.exists(self.path):
            return 0

        # weights_only=False because we also store the RNG states, which are not
        # plain tensors. These are our own files, written by `save()` below.
        ckpt = torch.load(self.path, map_location='cpu', weights_only=False)

        for name, values in ckpt.get('history', {}).items():
            if name in self.history:
                self.history[name][:] = list(values)

        for name, model in self.models.items():
            model.load_state_dict(ckpt['models'][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(ckpt['optimizers'][name])
        _set_rng_state(ckpt.get('rng'))

        self.next_epoch = int(ckpt.get('next_epoch', 0))
        self.completed = bool(ckpt.get('completed', False))

        if self.completed:
            print(f'[skip] {self.path}: already complete.')
        else:
            print(f'[resume] {self.path}: continuing from epoch {self.next_epoch + 1}.')
        return self.next_epoch

    def save(self, next_epoch: int):
        """Call at the end of every epoch, passing the *next* epoch index."""
        if not self.enabled:
            return
        _atomic_torch_save({
            'next_epoch': int(next_epoch),
            'completed': False,
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'history': {name: list(values) for name, values in self.history.items()},
            'rng': _get_rng_state(),
        }, self.path)
        self.next_epoch = int(next_epoch)

    def finalise(self, final_path: Optional[str] = None, payload: Any = None):
        """Save the run's final artefact and mark the checkpoint as complete."""
        if final_path is not None:
            _atomic_torch_save(payload, final_path)
        if self.enabled:
            _atomic_torch_save({
                'next_epoch': self.next_epoch,
                'completed': True,
                'models': {name: model.state_dict() for name, model in self.models.items()},
                'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
                'history': {name: list(values) for name, values in self.history.items()},
                'rng': _get_rng_state(),
            }, self.path)
        self.completed = True


class ResultsStore:
    """
    Crash-safe store for the per-run result curves used to build the final
    figure (e.g. V(S_0) at every epoch, for every run). It is rewritten
    atomically after each completed run, so an interrupted sweep resumes with
    the runs that already finished.
    """

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.run_ids: List[str] = []
        self.runs: Dict[str, Dict[str, np.ndarray]] = {}
        self.meta: Dict[str, np.ndarray] = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        with np.load(self.path, allow_pickle=False) as data:
            self.run_ids = [str(run_id) for run_id in data['__run_ids__']]
            self.runs = {run_id: {} for run_id in self.run_ids}
            self.meta = {}
            for key in data.files:
                if key == '__run_ids__':
                    continue
                owner, _, name = key.partition('|')
                if owner == 'meta':
                    self.meta[name] = data[key]
                else:
                    self.runs.setdefault(owner, {})[name] = data[key]
        if self.enabled:
            print(f'[results] {self.path}: {len(self.run_ids)} completed run(s) loaded.')

    def has(self, run_id: str) -> bool:
        return run_id in self.runs

    def add(self, run_id: str, **curves):
        """Record (or overwrite) one run's curves and flush to disk immediately."""
        if run_id not in self.runs:
            self.run_ids.append(run_id)
        self.runs[run_id] = {k: np.asarray(v, dtype=np.float64) for k, v in curves.items()}
        self.flush()

    def set_meta(self, **values):
        self.meta.update({k: np.asarray(v, dtype=np.float64) for k, v in values.items()})
        self.flush()

    def stack(self, name: str) -> np.ndarray:
        """Curves for `name` across every completed run, shaped (n_runs, n_epochs)."""
        return np.stack([self.runs[run_id][name] for run_id in self.run_ids])

    def flush(self):
        if not self.enabled:
            return
        payload = {'__run_ids__': np.array(self.run_ids, dtype='<U64')}
        for run_id in self.run_ids:
            for name, values in self.runs[run_id].items():
                payload[f'{run_id}|{name}'] = values
        for name, values in self.meta.items():
            payload[f'meta|{name}'] = values

        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        # Dot-prefixed, so a temporary file left behind by a crash is not picked
        # up by the shard glob below.
        tmp_path = os.path.join(directory, f'.{os.path.basename(self.path)}.tmp.npz')
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, self.path)


def completed_run_ids(pattern: str) -> set:
    """
    Every run id already recorded in any store matching `pattern`. Used so that
    parallel workers (and relaunches) never redo a run another worker finished.
    """
    ids = set()
    for path in sorted(glob.glob(pattern)):
        ids.update(ResultsStore(path, enabled=False).run_ids)
    return ids


def merge_result_shards(pattern: str, out_path: str) -> ResultsStore:
    """
    Combine the per-worker result files written by a parallel sweep into a
    single store. Runs are keyed by id, so re-merging is idempotent. If only one
    file matches there is nothing to merge and it is returned directly.
    """
    sources = [path for path in sorted(glob.glob(pattern))
               if os.path.abspath(path) != os.path.abspath(out_path)]
    if not sources:
        raise FileNotFoundError(f'No FQE result files matched {pattern!r}. Has the sweep been run?')
    if len(sources) == 1:
        return ResultsStore(sources[0])

    merged = ResultsStore(out_path)
    for path in sources:
        shard = ResultsStore(path, enabled=False)
        merged.meta.update(shard.meta)
        for run_id in shard.run_ids:
            if run_id not in merged.runs:
                merged.run_ids.append(run_id)
            merged.runs[run_id] = shard.runs[run_id]
    merged.flush()
    print(f'[results] merged {len(sources)} shard(s) -> {out_path} ({len(merged.run_ids)} run(s)).')
    return merged


# =============================================================================
#   AGGREGATION (Agarwal et al., 2021 - "Deep RL at the Edge of the
#   Statistical Precipice", NeurIPS 2021)
# =============================================================================
def iqm(scores: Union[np.ndarray, Sequence[float]], axis: int = 0) -> np.ndarray:
    """
    Interquartile mean: the mean of the middle 50% of the runs. This matches
    `scipy.stats.trim_mean(scores, proportiontocut=0.25)`, but is implemented
    here with numpy alone to avoid adding a dependency, and is vectorised so it
    can be applied to many bootstrap replicates at once.
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64), axis=axis)
    n_runs = scores.shape[axis]
    lower_cut = int(0.25 * n_runs)
    upper_cut = n_runs - lower_cut
    trimmed = np.take(scores, np.arange(lower_cut, upper_cut), axis=axis)
    return trimmed.mean(axis=axis)


def cluster_bootstrap_iqm_ci(
        runs: Union[np.ndarray, Sequence[float]],
        policy_seeds: Union[np.ndarray, Sequence[int]],
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 0,
        chunk_size: int = 1_000,
):
    """
    Two-stage hierarchical cluster percentile bootstrap confidence interval 
    for the interquartile mean.
    
    `runs` is either a 1-D array of per-run scores, or a 2-D array of per-run
    curves shaped (n_runs, n_epochs). 
    
    This interval resamples policy seeds with replacement first, then resamples 
    the runs within each draw to propagate the number of independent policies 
    into the interval.
    """
    runs = np.asarray(runs, dtype=np.float64)
    is_scalar = runs.ndim == 1
    if is_scalar:
        runs = runs[:, None]

    n_runs = runs.shape[0]
    rng = np.random.default_rng(seed)
    
    # Calculate the point estimate
    point = iqm(runs, axis=0)

    policy_seeds = np.asarray(policy_seeds)
    if policy_seeds.shape[0] != n_runs:
        raise ValueError(f'policy_seeds has {policy_seeds.shape[0]} entries but there '
                         f'are {n_runs} runs.')

    members = [np.flatnonzero(policy_seeds == seed_value)
               for seed_value in np.unique(policy_seeds)]
    n_groups = len(members)
    sizes = {group.size for group in members}

    if len(sizes) == 1:
        # Balanced design (the usual case): draw the whole index matrix at once.
        groups = np.stack(members)                       # (n_groups, group_size)
        group_size = groups.shape[1]

        def cluster_draw(batch):
            drawn = rng.integers(0, n_groups, size=(batch, n_groups))
            within = rng.integers(0, group_size, size=(batch, n_groups, group_size))
            picked = np.take_along_axis(groups[drawn], within, axis=2)
            return picked.reshape(batch, n_runs)
            
        replicates = []
        remaining = n_bootstrap
        while remaining > 0:
            batch = min(chunk_size, remaining)
            replicates.append(iqm(runs[cluster_draw(batch)], axis=1))
            remaining -= batch
        replicates = np.concatenate(replicates, axis=0)
        
        lower, upper = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
        
    else:
        # Unbalanced: a resample's size varies with which policies were drawn,
        # so the replicates are built one at a time.
        def cluster_draw(batch):
            return [np.concatenate([rng.choice(members[g], size=members[g].size, replace=True)
                                    for g in rng.integers(0, n_groups, size=n_groups)])
                    for _ in range(batch)]

        replicates = np.stack([iqm(runs[picked], axis=0)
                               for picked in cluster_draw(n_bootstrap)])
        lower, upper = np.percentile(
            replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)

    if is_scalar:
        return float(point[0]), float(lower[0]), float(upper[0])
    return point, lower, upper


def paired_differences(results_store, smdp_key, mdp_key, epoch=-1):
    """
    Per-run differences SMDP - MDP at `epoch`, with the policy seed each run
    belongs to. Returns (diffs, policy_seeds, smdp, mdp), all aligned by run.
    """
    diffs, policy_seeds, smdp, mdp = [], [], [], []
    for run_id in results_store.run_ids:
        match = re.match(r'p(\d+)_f(\d+)$', run_id)
        if match is None:
            raise ValueError(f'Run id {run_id!r} is not of the form p<policy>_f<fqe>; '
                             f'the paired analysis needs the multi-seed sweep.')
        run = results_store.runs[run_id]
        a, b = float(run[smdp_key][epoch]), float(run[mdp_key][epoch])
        smdp.append(a)
        mdp.append(b)
        diffs.append(a - b)
        policy_seeds.append(int(match.group(1)))
    return (np.array(diffs), np.array(policy_seeds), np.array(smdp), np.array(mdp))


# ---------------------------------------------------------------------------
#   Paired cluster bootstrap 95% CI on the IQM of the per-run differences
# ---------------------------------------------------------------------------
def paired_bootstrap_iqm_diff_ci(diffs, policy_seeds, n_bootstrap=10_000,
                                 alpha=0.05, seed=0):
    """
    Percentile bootstrap CI for IQM(SMDP - MDP), resampling whole matched runs.

    This resamples the 5 policy seeds with replacement and then the 5 FQE runs 
    within each drawn seed, which respects the fact that five runs share a policy 
    checkpoint and so are not independent.
    """
    diffs = np.asarray(diffs, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    if policy_seeds is None:
        raise ValueError('Cluster bootstrap requires policy_seeds.')
        
    groups = [np.flatnonzero(policy_seeds == s) for s in np.unique(policy_seeds)]
    n_groups = len(groups)
    replicates = np.empty(n_bootstrap)
    
    for r in range(n_bootstrap):
        drawn = rng.integers(0, n_groups, size=n_groups)
        picked = np.concatenate([rng.choice(groups[g], size=groups[g].size, replace=True)
                                 for g in drawn])
        replicates[r] = iqm(diffs[picked])

    lower, upper = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        'n_pairs': diffs.size,
        'iqm_diff': float(iqm(diffs)),
        'mean_diff': float(diffs.mean()),
        'ci_lower': float(lower), 'ci_upper': float(upper),
        'boot_se': float(replicates.std(ddof=1)),
    }


def report_bootstrap_diff(results_store, algo_pairs, epoch=-1, n_bootstrap=10_000, alpha=0.05, seed=0):
    """
    Table of IQM(SMDP - MDP) with a paired bootstrap CI, one row per algorithm,
    using every run's V(S_0) at `epoch` (default: the final epoch).
    """
    rows = {}
    scheme = 'resampling policy seeds, then runs within them'
    
    print(f'IQM of the per-run difference (SMDP - MDP) at epoch index {epoch}, with a '
          f'{100 * (1 - alpha):.0f}% percentile bootstrap CI\n'
          f'({n_bootstrap:,} replicates, paired: {scheme}).')
    print(f'{"Algorithm":<12}{"pairs":>7}{"IQM diff":>11}{"mean diff":>11}{"95% CI":>24}')
    
    for algo, (smdp_key, mdp_key) in algo_pairs.items():
        diffs, policy_seeds, _, _ = paired_differences(results_store, smdp_key, mdp_key, epoch)
        res = paired_bootstrap_iqm_diff_ci(diffs, policy_seeds, n_bootstrap=n_bootstrap,
                                           alpha=alpha, seed=seed)
        rows[algo] = res
        interval = f'[{res["ci_lower"]:.4f}, {res["ci_upper"]:.4f}]'
        print(f'{algo:<12}{res["n_pairs"]:>7}{res["iqm_diff"]:>11.4f}'
              f'{res["mean_diff"]:>11.4f}{interval:>24}')
    return rows


# ---------------------------------------------------------------------------
#   Per-policy-seed paired t-tests with Benjamini-Hochberg correction
# ---------------------------------------------------------------------------
def paired_diff_test(diffs):
    """Mean of the 5 matched differences, its SE, and the paired t-test."""
    d = np.asarray(diffs, dtype=float)
    test = stats.ttest_1samp(d, popmean=0.0)        # == paired t-test, two-sided
    return {'n_pairs': d.size, 'mean_diff': float(d.mean()),
            'se': float(d.std(ddof=1) / np.sqrt(d.size)),
            't': float(test.statistic), 'df': float(d.size - 1), 'p': float(test.pvalue)}


def report_per_policy_seed(results_store, algo_pairs, epoch=-1, bh_family='algorithm'):
    """
    Per policy seed and algorithm: the mean paired SMDP - MDP difference over
    that seed's 5 matched FQE runs, its standard error, the paired t-test
    p-value, and the BH-adjusted q-value.

    bh_family='algorithm' corrects across the 5 policy seeds within each
    algorithm; bh_family='all' corrects across all 10 tests at once.
    """
    rows = []
    for algo, (smdp_key, mdp_key) in algo_pairs.items():
        diffs, policy_seeds, _, _ = paired_differences(results_store, smdp_key, mdp_key, epoch)
        for policy_seed in np.unique(policy_seeds):
            row = {'algorithm': algo, 'policy_seed': int(policy_seed)}
            row.update(paired_diff_test(diffs[policy_seeds == policy_seed]))
            rows.append(row)

    if bh_family == 'all':
        families = [rows]
    else:
        families = [[row for row in rows if row['algorithm'] == algo] for algo in algo_pairs]
    for family in families:
        q = stats.false_discovery_control([row['p'] for row in family], method='bh')
        for row, value in zip(family, q):
            row['q'] = float(value)

    print(f'\nPer policy seed, paired SMDP - MDP at epoch index {epoch} '
          f'(paired t-test, BH correction across {bh_family}):')
    print(f'{"Algorithm":<11}{"seed":>5}{"pairs":>7}{"diff":>10}{"SE":>9}'
          f'{"t":>8}{"df":>5}{"p":>10}{"q (BH)":>10}')
    for row in rows:
        print(f'{row["algorithm"]:<11}{row["policy_seed"]:>5}{row["n_pairs"]:>7}'
              f'{row["mean_diff"]:>10.4f}{row["se"]:>9.4f}{row["t"]:>8.2f}'
              f'{row["df"]:>5.0f}{row["p"]:>10.4f}{row["q"]:>10.4f}')
    return rows