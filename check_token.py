import json

json_path = "/home/wuroderi/scratch/reasoning_traces/Qwen2.5-32B/velocity/traces_metadata.json"

traces = json.load(open(json_path, "r"))

print(traces[4]["token_strings"][80:100]) #90, 95, 91, 88, 93