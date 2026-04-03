from __future__ import annotations

from typing import Iterable

import torch


class FIFOStreamMemory:
    """Per-batch FIFO memory queue for one stream."""

    def __init__(self, max_steps: int, stop_grad: bool = True) -> None:
        if max_steps <= 0:
            raise ValueError(f'max_steps must be > 0, got {max_steps}')
        self.max_steps = int(max_steps)
        self.stop_grad = bool(stop_grad)
        self._tokens: list[torch.Tensor] = []
        self._valid: list[torch.Tensor] = []

    def reset(self) -> None:
        self._tokens.clear()
        self._valid.clear()

    def __len__(self) -> int:
        return len(self._tokens)

    def append(self, tokens: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
        """
        Args:
            tokens: [B, T, C]
            valid_mask: [B] bool, True means this timestep is valid for that sample.
        """
        if tokens.ndim != 3:
            raise ValueError(f'tokens must be [B,T,C], got {tuple(tokens.shape)}')

        b = tokens.shape[0]
        device = tokens.device
        if valid_mask is None:
            valid_mask = torch.ones(b, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        x = tokens.detach() if self.stop_grad else tokens
        x = x * valid_mask[:, None, None].to(dtype=x.dtype)

        self._tokens.append(x)
        self._valid.append(valid_mask)

        if len(self._tokens) > self.max_steps:
            self._tokens.pop(0)
            self._valid.pop(0)

    def gather(
        self,
        current_tokens: torch.Tensor,
        current_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            kv_tokens: [B, N, C]
            key_padding_mask: [B, N] with True as masked positions.
        """
        if current_tokens.ndim != 3:
            raise ValueError(f'current_tokens must be [B,T,C], got {tuple(current_tokens.shape)}')
        if current_valid.ndim != 1:
            raise ValueError(f'current_valid must be [B], got {tuple(current_valid.shape)}')

        tokens_seq: list[torch.Tensor] = [*self._tokens, current_tokens]
        valid_seq: list[torch.Tensor] = [*self._valid, current_valid.to(dtype=torch.bool)]

        token_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        for tok, v in zip(tokens_seq, valid_seq):
            # Per-token validity for one memory step.
            step_valid = v[:, None].expand(tok.shape[0], tok.shape[1])
            token_chunks.append(tok)
            mask_chunks.append(~step_valid)

        kv_tokens = torch.cat(token_chunks, dim=1)
        key_padding_mask = torch.cat(mask_chunks, dim=1)
        return kv_tokens, key_padding_mask


class StreamMemoryBundle:
    """Dual-stream action/scene memory container."""

    def __init__(self, action_len: int, scene_len: int, stop_grad: bool = True) -> None:
        self.action = FIFOStreamMemory(max_steps=action_len, stop_grad=stop_grad)
        self.scene = FIFOStreamMemory(max_steps=scene_len, stop_grad=stop_grad)

    def reset(self) -> None:
        self.action.reset()
        self.scene.reset()
