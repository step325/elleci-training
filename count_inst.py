import re

with open('new_istruction.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Count instruction fields
matches = re.findall(r'"instruction":', content)
print(f'Found {len(matches)} instruction fields')

# Count closing braces at start of lines
braces = re.findall(r'^\{', content, re.MULTILINE)
print(f'Found {len(braces)} opening braces at line start')
