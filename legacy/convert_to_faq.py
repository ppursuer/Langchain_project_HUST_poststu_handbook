import json

input_file = r'c:\Users\x1359\Desktop\rag_project\datas\lora_hust_student_handbookt.jsonl'
output_file = r'c:\Users\x1359\Desktop\rag_project\datas\lora_hust_student_handbookt_faq.json'

faq_list = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        conversations = data.get('conversations', [])
        
        query = None
        answer = None
        
        for conv in conversations:
            if conv.get('role') == 'user':
                query = conv.get('content')
            elif conv.get('role') == 'assistant':
                answer = conv.get('content')
        
        if query and answer:
            faq_list.append({
                "query": query,
                "answer": answer
            })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(faq_list, f, ensure_ascii=False, indent=2)

print(f"转换完成！共 {len(faq_list)} 条FAQ数据")
print(f"输出文件: {output_file}")
