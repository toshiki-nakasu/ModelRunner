import torch

# PyTorchバージョンとCUDAサポートを確認
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 簡単な操作でGPUが機能することを確認
    x = torch.rand(5, 3).cuda()
    print(f"Tensor on GPU: {x}")

if hasattr(torch.backends, 'cudnn'):
    print(f"CuDNN version: {torch.backends.cudnn.version()}")
    print(f"CuDNN enabled: {torch.backends.cudnn.enabled}")
