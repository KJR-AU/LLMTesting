from agent import agent
print("Ask questions about Uber's 10-k filings between 2019 and 2022. Type 'exit' to exit.")
while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")