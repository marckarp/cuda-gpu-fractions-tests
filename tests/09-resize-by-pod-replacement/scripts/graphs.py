#!/usr/bin/env python3
"""Filter: read a pod from stdin, drop --enforce-eager (so vLLM captures CUDA graphs) and
raise --gpu-memory-utilization a little to pay for the ~0.56 GiB the graphs take.
    mkpod.py prefill prefill-g 0.6 | graphs.py 0.34 | kubectl apply -f -
"""
import sys

import yaml

util = sys.argv[1]
pod = yaml.safe_load(sys.stdin)
args = pod["spec"]["containers"][0]["args"]
args.remove("--enforce-eager")
args[args.index("--gpu-memory-utilization") + 1] = util
print(yaml.safe_dump(pod))
