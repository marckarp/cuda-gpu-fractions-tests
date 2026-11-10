import json,urllib.request,time
body={"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Summarize: "+"the quick brown fox jumps over the lazy dog. "*350}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":False}}
for i in range(3):
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=120)
    u=json.load(r)["usage"]; print(f"long request ok: prompt_tokens={u['prompt_tokens']} completion={u['completion_tokens']} in {time.time()-t:.2f}s")
