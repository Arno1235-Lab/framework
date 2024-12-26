print('---')
print('pytorch')
try:
    import torch
    print(f'gpu available: {torch.cuda.is_available()}')
    print(f'amount of cuda devices: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        print(f'version: {torch.version.cuda}')
        for i in range(torch.cuda.device_count()):
            print(f'device name: {torch.cuda.get_device_name(i)}')
except:
    print("pytorch is not installed")

print('---')
print("tensorflow")
try:
    import tensorflow as tf

    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
except:
    print("tensorflow is not installed")

print('---')
print('tinygrad')
try:
    import tinygrad.tensor as T

    try:
        # Attempt to create a CUDA tensor filled with zeros
        cuda_tensor = T.Tensor.zeros((1,), device='cuda')
        print("CUDA is available!")
    except RuntimeError as e:
        print("CUDA is not available:", e)

except:
    print("tinygrad is not installed")

print('---')
