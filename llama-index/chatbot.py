from agent import agent
import os
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

years = config['years']
min_yr = min(years)
max_yr = max(years)

print(f"Ask questions about Uber's 10-k filings between {min_yr} and {max_yr}. Type 'exit' to exit.")
while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")