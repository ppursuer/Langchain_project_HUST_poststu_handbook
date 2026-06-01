import json
import re

# Read level.json
with open(r'C:\Users\x1359\Desktop\rag_project\datas\level.json', 'r', encoding='utf-8') as f:
    level_data = json.load(f)

# Read full.md
with open(r'c:\Users\x1359\Desktop\rag_project\datas\full.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Build a list of document titles
all_titles = []
title_to_category = {}
for category, titles in level_data.items():
    for title in titles:
        all_titles.append(title)
        title_to_category[title] = category

# Split content into lines
lines = content.split('\n')

# Find where actual content starts (after table of contents)
content_start_idx = None
for i, line in enumerate(lines):
    stripped = line.strip()
    # Check if this line matches any document title
    if stripped in all_titles:
        content_start_idx = i
        break

print(f"Content starts at line: {content_start_idx}")

# Process lines from content_start_idx
result_lines = []
current_category = None
i = content_start_idx if content_start_idx else 0

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Check if this line is a document title
    matched_title = None
    for title in all_titles:
        if stripped == title:
            matched_title = title
            break
    
    if matched_title:
        category = title_to_category[matched_title]
        # Add level 1 heading (category) if it changed
        if current_category != category:
            current_category = category
            # Add empty line before category heading
            if result_lines and result_lines[-1].strip():
                result_lines.append('')
            result_lines.append(f'# {current_category}')
            result_lines.append('')
        
        # Add level 2 heading for document title
        result_lines.append(f'## {matched_title}')
        result_lines.append('')
        i += 1
        continue
    
    # Process "第x章" to level 3 heading with newline after
    chapter_match = re.match(r'^(第[一二三四五六七八九十百千\d]+章)(.*)', stripped)
    if chapter_match:
        chapter_num = chapter_match.group(1)
        chapter_content = chapter_match.group(2).strip()
        result_lines.append(f'### {chapter_num} {chapter_content}')
        result_lines.append('')
        i += 1
        continue
    
    # Process "第x条" to level 4 heading with newline after
    article_match = re.match(r'^(第[一二三四五六七八九十百千\d]+条)(.*)', stripped)
    if article_match:
        article_num = article_match.group(1)
        article_content = article_match.group(2).strip()
        result_lines.append(f'#### {article_num}')
        result_lines.append(article_content)
        i += 1
        continue
    
    result_lines.append(line)
    i += 1

# Write result to full.md
with open(r'c:\Users\x1359\Desktop\rag_project\datas\full.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(result_lines))

print(f"Output file length: {len(result_lines)} lines")
print("Processing complete!")
