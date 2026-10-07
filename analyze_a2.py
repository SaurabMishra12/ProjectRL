import json
file_path = 'c:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\08-06results\\continuous\\Toy_GSM8K_gsm8k_stream.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        a2_init = str(data['initial_answers']['2']).strip()
        a2_final = str(data['final_answers']['2']).strip()
        gold = str(data['gold_answer']).strip()
        if data['initial_correct']['2'] and a2_final != gold:
            print(f"Question: {data['question']}")
            print(f"Gold: {gold}")
            print(f"Agent 2 Initial Answer: {a2_init}")
            print(f"Agent 2 Final Answer:   {a2_final}")
            print(f"Agent 2 Final Reasoning:\n{data['final_reasoning']['2']}\n---")
