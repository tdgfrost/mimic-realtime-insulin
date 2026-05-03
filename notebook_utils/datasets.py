import os
from typing import Dict, List
import torch
from safetensors import safe_open


class _BaseDataset:
    """
    GPU-resident dataset that loads all safetensor data onto the target device
    at initialisation. Subclasses precompute the concatenated tensors they need
    and discard the raw data to minimise GPU memory usage.
    """
    def __init__(self, segment_dir: str, device: str = 'cuda'):
        self.segment_dir = segment_dir
        self.device = device
        self.tensor_names = [
            'states', 'next_states', 'actions', 'next_actions',
            'dones', 'reward_markers', 'infos'
        ]
        self.state_keys = [
            'feature', 'time', 'value'
        ]
        self.action_keys = [
            'insulin_maintain', 'insulin_stop', 'insulin_change', 'insulin_delta_change'
        ]
        self.reward_marker_keys = [
            'next_bm', '1-day-alive', '1-day-alive-final', '3-day-alive', '3-day-alive-final', '7-day-alive', '7-day-alive-final', 
            '14-day-alive', '14-day-alive-final', '28-day-alive', '28-day-alive-final'
        ]
        self.info_keys = [
            'current_bm', 'prev_bm', 'episode_num', 'step_num', 'insulin_old_rate', 'insulin_new_rate', 'label_id', 'minutes_remaining', 
            'steps_per_episode', 'steps_remaining', 'time_since_prev_bm', 'time_until_next_bm'
        ]

        # Load all raw data onto device
        self.raw_data = self._load_all_data()
        self.total_rows = self.raw_data['dones']['is_done'].shape[0]
        self.seq_len = self.raw_data['states']['value'].shape[1] if 'states' in self.raw_data else None

        # Let subclass precompute what it needs, then discard raw data
        self._precompute()

    def _load_all_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load all safetensor files eagerly onto the target device."""
        data = {}
        for tensor_name in self.tensor_names:
            filepath = os.path.join(self.segment_dir, f'{tensor_name}.safetensors')
            if not os.path.exists(filepath):
                continue
            data[tensor_name] = {}
            with safe_open(filepath, framework='pt', device='cpu') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if self.device == 'cuda':
                        tensor = tensor.pin_memory().to(self.device, non_blocking=True)
                    else:
                        tensor = tensor.to(self.device)
                    data[tensor_name][key] = tensor
        # Synchronise to ensure all transfers are complete
        if self.device == 'cuda':
            torch.cuda.synchronize()
        return data

    def _precompute(self):
        """Override in subclasses to precompute and store only what's needed."""
        pass

    def _concat_tensors(self, key: str, all_keys: List[str]):
        return torch.cat([self.raw_data[key][k] for k in all_keys], dim=-1)

    def _concat_states(self, override_keys: List[str] = None, is_next: bool = False):
        """Concatenate state keys into a single [N, seq_len, 3] tensor."""
        if override_keys is not None:
            self.state_keys = override_keys
        prefix = 'next_' if is_next else ''
        return self._concat_tensors(f'{prefix}states', self.state_keys)

    def _concat_actions(self, override_keys: List[str] = None, is_next: bool = False):
        """Concatenate action keys into a single [N, 4] tensor."""
        if override_keys is not None:
            self.action_keys = override_keys
        prefix = 'next_' if is_next else ''
        return self._concat_tensors(f'{prefix}actions', self.action_keys)

    def _concat_reward_markers(self, override_keys: List[str] = None):
        """Concatenate action keys into a single [N, 11] tensor."""
        if override_keys is not None:
            self.reward_marker_keys = override_keys
        return self._concat_tensors('reward_markers', self.reward_marker_keys)

    def _concat_infos(self, override_keys: List[str] = None):
        """Concatenate action keys into a single [N, 12] tensor."""
        if override_keys is not None:
            self.info_keys = override_keys
        return self._concat_tensors('infos', self.info_keys)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __getitem__")



class ExampleDataset(_BaseDataset):
    def __init__(self, segment_dir: str, device: str = 'cuda'):
        super().__init__(segment_dir, device)

    def _precompute(self):
        self.states = self._concat_states()
        self.actions = self._concat_actions()
        self.reward_markers = self._concat_reward_markers()
        self.next_states = self._concat_states(is_next=True)
        self.next_actions = self._concat_actions(is_next=True)
        self.infos = self._concat_infos()
        self.dones = self.raw_data['dones']['is_done']

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'reward_markers': self.reward_markers[idx],
            'next_states': self.next_states[idx],
            'next_actions': self.next_actions[idx],
            'dones': self.dones[idx],
            'infos': self.infos[idx],
        }


class CloningDataset(_BaseDataset):
    def __init__(self, segment_dir: str, device: str = 'cuda'):
        super().__init__(segment_dir, device)

    def _precompute(self):
        # Information used for UMAP later on
        self.insulin_delta_change = self._concat_actions(override_keys=['insulin_delta_change'])
        self.current_bm = self._concat_infos(override_keys=['current_bm'])
        self.episode_num = self._concat_infos(override_keys=['episode_num'])
        self.three_day_alive = self._concat_reward_markers(override_keys=['3-day-alive'])
        self.insulin_old_rate = self._concat_infos(override_keys=['insulin_old_rate'])
        self.minutes_remaining = self._concat_infos(override_keys=['minutes_remaining'])
        self.time_until_next_bm = self._concat_infos(override_keys=['time_until_next_bm'])

        # States/actions for training
        self.states = self._concat_states()
        self.actions = self._concat_actions(override_keys=['insulin_maintain', 'insulin_stop', 'insulin_change'])

        delta_change = self._concat_actions(override_keys=['insulin_delta_change']).squeeze(-1)

        self.valid_doses = torch.tensor(
            [-5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 5],
            device=self.device
        )

        diffs = torch.abs(delta_change.unsqueeze(1) - self.valid_doses.unsqueeze(0))
        dose_actions = torch.argmin(diffs, dim=1)
        self.actions = torch.cat((self.actions, dose_actions.view(-1, 1)), dim=-1).long()

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'infos': {'current_bm': self.current_bm[idx],
                      'episode_num': self.episode_num[idx],
                      'insulin_delta_change': self.insulin_delta_change[idx],
                      '3-day-alive': self.three_day_alive[idx],
                      'insulin_old_rate': self.insulin_old_rate[idx],
                      'minutes_remaining': self.minutes_remaining[idx],
                      'time_until_next_bm': self.time_until_next_bm[idx]},
        }


class FQEDataset(_BaseDataset):
    def __init__(self, segment_dir: str, device: str = 'cuda'):
        super().__init__(segment_dir, device)

    def _precompute(self):
        self.states = self._concat_states()
        self.actions = self._concat_actions(override_keys=['insulin_maintain', 'insulin_stop', 'insulin_change'])
        self.reward_markers = self._concat_reward_markers(override_keys=['next_bm'])
        # The overrides above are permanent and will carry across to the next states/actions
        self.next_states = self._concat_states(is_next=True)
        self.next_actions = self._concat_actions(is_next=True)
        self.time_until_next_bm = self._concat_infos(override_keys=['time_until_next_bm'])
        self.dones = self.raw_data['dones']['is_done']

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'reward_markers': self.reward_markers[idx],
            'next_states': self.next_states[idx],
            'next_actions': self.next_actions[idx],
            'dones': self.dones[idx],
            'infos': {'time_until_next_bm': self.time_until_next_bm[idx]},
        }
