import io

import torch as th
import torch.distributed as dist


def dev():
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load checkpoint and normalize common nesting keys.
    """
    if not dist.is_available() or not dist.is_initialized():
        state_dict = th.load(path, **kwargs)
    else:
        chunk_size = 2 ** 30
        if dist.get_rank() == 0:
            with open(path, "rb") as f:
                data = f.read()
            num_chunks = (len(data) + chunk_size - 1) // chunk_size
            num_chunks_tensor = th.tensor([num_chunks], dtype=th.int64)
            dist.broadcast(num_chunks_tensor, 0)
            for i in range(0, len(data), chunk_size):
                chunk = th.tensor(list(data[i:i + chunk_size]), dtype=th.uint8)
                dist.broadcast(chunk, 0)
        else:
            num_chunks_tensor = th.tensor([0], dtype=th.int64)
            dist.broadcast(num_chunks_tensor, 0)
            num_chunks = num_chunks_tensor.item()
            data = bytearray()
            for _ in range(num_chunks):
                chunk = th.zeros(chunk_size, dtype=th.uint8)
                dist.broadcast(chunk, 0)
                data.extend(chunk.tolist())
        state_dict = th.load(io.BytesIO(data), **kwargs)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
    return state_dict


def sync_params(params):
    if not dist.is_available() or not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)
