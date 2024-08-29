import torch
# from torch import __version__; from packaging.version import Version as V
# xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
#
# print(V(__version__))
#
# print(xformers)
#
# cmd = f"pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton"
# print(cmd)


print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(1))    # 根据索引号得到GPU名称