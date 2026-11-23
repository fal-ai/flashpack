"""Reclaim unused regions of shared model storage after replacing layers."""

from collections import defaultdict

import torch
from torch import nn

_POINTER_ALIGNMENT = 256


@torch.inference_mode(False)
@torch.no_grad()
def compact(model: nn.Module) -> int:
    """Compact plain registered tensors and return their storage reduction in bytes.

    Preserves values, strides, overlapping views, tied tensors and alignment.
    Call after replacing layers, before compilation, optimizer creation or saving
    external tensor views. Retained FlashPack load references are discarded.
    """
    groups = defaultdict(list)
    for child in model.modules():
        for table in (child._parameters, child._buffers):
            for name, tensor in table.items():
                if (
                    type(tensor) not in (torch.Tensor, nn.Parameter)
                    or tensor.is_meta
                    or tensor.layout != torch.strided
                ):
                    continue
                storage = tensor.untyped_storage()
                start = tensor.storage_offset() * tensor.element_size()
                span = (
                    1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
                    if tensor.numel()
                    else 0
                ) * tensor.element_size()
                groups[tensor.device, storage.data_ptr()].append(
                    (table, name, tensor, start, start + span)
                )

    for child in model.modules():
        child.__dict__.pop("_flash_shared_storage", None)
        child.__dict__.pop("_flash_shared_storage_meta", None)
    released = 0
    for (device, _), entries in groups.items():
        storage = entries[0][2].untyped_storage()
        intervals = []
        for start, end in sorted(
            {(start, end) for _, _, _, start, end in entries if end > start}
        ):
            if intervals and start <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])
        capacity = sum(end - start for start, end in intervals)
        capacity += _POINTER_ALIGNMENT * len(intervals)
        if capacity >= storage.nbytes():
            continue

        arena = torch.empty(capacity, dtype=torch.uint8, device=device)
        source = torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        mapping = []
        cursor = 0
        for start, end in intervals:
            cursor += (
                storage.data_ptr() + start - arena.data_ptr() - cursor
            ) % _POINTER_ALIGNMENT
            arena[cursor : cursor + end - start].copy_(source[start:end])
            mapping.append((start, end, cursor))
            cursor += end - start
        replacements = {}
        for table, name, tensor, start, end in entries:
            identity = id(tensor)
            if identity in replacements:
                table[name] = replacements[identity]
                continue
            offset = 0
            if tensor.numel():
                left, _, destination = next(
                    entry for entry in mapping if entry[0] <= start and end <= entry[1]
                )
                offset = (destination + start - left) // tensor.element_size()
            copied = torch.empty(0, dtype=tensor.dtype, device=device).set_(
                arena.untyped_storage(), offset, tensor.shape, tensor.stride()
            )
            if tensor.is_conj():
                copied = copied.conj()
            if tensor.is_neg():
                copied = copied._neg_view()
            replacements[identity] = (
                nn.Parameter(copied, requires_grad=tensor.requires_grad)
                if isinstance(tensor, nn.Parameter)
                else copied.requires_grad_(tensor.requires_grad)
            )
            table[name] = replacements[identity]
        released += storage.nbytes() - arena.untyped_storage().nbytes()

        entries.clear()
        del source, storage, tensor
    return released
