import time, torch
t=time.time()
def s(m): print(f"{time.time()-t:5.1f}s {m}", flush=True)
s("torch imported"); torch.cuda.init(); s("cuda init")
p=torch.cuda.get_device_properties(0); f,tot=torch.cuda.mem_get_info(); s(f"sm={p.multi_processor_count} total={tot>>20}MiB")
n=8192; a=torch.randn(n,n,device="cuda",dtype=torch.float16); b=torch.randn(n,n,device="cuda",dtype=torch.float16); torch.cuda.synchronize(); s("tensors")
t1=time.time(); it=0
while time.time()-t1<3: c=a@b; torch.cuda.synchronize(); it+=1
s(f"{it*2*n**3/(time.time()-t1)/1e12:.1f} TFLOP/s")
import os
mib=int(os.environ.get("PROBE_MIB","0"))
if mib:
    try:
        x=torch.empty(mib*1024*1024,dtype=torch.uint8,device="cuda"); s(f"alloc {mib}MiB ok")
    except Exception as e: s(f"alloc {mib}MiB FAIL")
