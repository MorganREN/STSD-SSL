_base_ = ["../../../third_party/Pointcept/configs/_base_/default_runtime.py"]

# Hardware-independent batching (resolved at launch by stsd.engines.resolve_batch):
# per-GPU batch and gradient accumulation follow from the world size.
global_batch_size = 24
max_batch_size_per_gpu = 8
num_worker_per_gpu_target = 12
sync_bn = "auto"  # SyncBN when >1 GPU and <=4 samples per GPU

mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
enable_wandb = False
wandb_project = "stsd-ssl"
seed = 0
