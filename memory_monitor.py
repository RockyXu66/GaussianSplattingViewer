from nvitop import Device
import psutil
import time
from loguru import logger

def get_pid_by_script_name(script_name):
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the script name is in the command line arguments
            if script_name in proc.info['cmdline']:
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

script_name = 'main.py'
pid = get_pid_by_script_name(script_name)
if pid:
    print(f"The PID of '{script_name}' is {pid}")
else:
    print(f"No process found running with the script name '{script_name}'")

device = Device.all()[0]  # or `Device.cuda.all()` to use CUDA ordinal instead

total_st = time.time()
st = time.time()
interval = 1 / 10
duration = 10
cnt = 0
total = 0

while True:
    if time.time() - total_st > duration:
        break
    if time.time() - st < interval:
        time.sleep(1e-3)
        continue
    else:
        st = time.time()
    processes = device.processes()  # type: Dict[int, GpuProcess]
    memory = processes[pid]._gpu_memory
    total += memory
    cnt += 1

logger.info(f'Average VRAM in {duration} seconds is {(total / cnt / 1024 / 1024):.0f} MB')

