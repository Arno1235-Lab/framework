
print('---')
print('pytorch')
import torch
print(f'gpu available: {torch.cuda.is_available()}')
print(f'amount of cuda devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'device name: {torch.cuda.get_device_name(i)}')

#print('---')
#print("tensorflow")
#import tensorflow as tf
#print(tf.test.is_gpu_available())
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print('---')
print('tinygrad')
import tinygrad.tensor as T

try:
    # Attempt to create a CUDA tensor filled with zeros
    cuda_tensor = T.Tensor.zeros((1,), device='cuda')
    print("CUDA is available!")
except RuntimeError as e:
    print("CUDA is not available:", e)

print('---')
