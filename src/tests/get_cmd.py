from torch import __version__; from packaging.version import Version as V
xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"

print(V(__version__))

print(xformers)

cmd = f"pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton"
print(cmd)