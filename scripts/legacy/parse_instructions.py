import json
import re

with open('new_istruction.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Split on lines that start with {
lines = content.split('\n')
objects = []
current_obj = []

for line in lines:
    stripped = line.strip()
    if stripped.startswith('{') and current_obj:
        # Start of new object, save previous
        obj_text = '\n'.join(current_obj)
        objects.append(obj_text)
        current_obj = [line]
    else:
        current_obj.append(line)

# Don't forget the last one
if current_obj:
    objects.append('\n'.join(current_obj))

print(f'Split into {len(objects)} potential objects')

# Parse each object with raw_decode for better error handling
instructions = []
decoder = json.JSONDecoder()

for i, obj_text in enumerate(objects):
    try:
        # First try direct parse
        obj = json.loads(obj_text)
        if 'instruction' in obj:
            instructions.append(obj)
    except json.JSONDecodeError:
        # Try to fix common issues
        fixed = obj_text
        # Fix unescaped backslashes before non-escape chars
        # Valid: \n \r \t \\ \" \/ \b \f \uXXXX
        # Replace \ before anything else with \\
        fixed = re.sub(r'\\(?![nrtbf\\/\"u])', r'\\\\', fixed)
        
        try:
            obj = json.loads(fixed)
            if 'instruction' in obj:
                instructions.append(obj)
        except json.JSONDecodeError as e:
            # Try another fix - escape quotes inside strings
            # This is complex, let's try a more aggressive approach
            try:
                # Use raw_string_decode pattern
                # Extract instruction and output manually
                inst_match = re.search(r'"instruction"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', obj_text)
                inp_match = re.search(r'"input"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', obj_text)
                out_match = re.search(r'"output"\s*:\s*"(.*)"', obj_text, re.DOTALL)
                
                if inst_match and out_match:
                    # Reconstruct manually
                    new_obj = {
                        'instruction': inst_match.group(1).replace('\\n', '\n'),
                        'input': inp_match.group(1) if inp_match else '',
                        'output': out_match.group(1).replace('\\n', '\n')
                    }
                    instructions.append(new_obj)
                else:
                    print(f'Object {i} failed all parsing attempts')
            except Exception as e2:
                print(f'Object {i} regex failed: {e2}')

print(f'\nSuccessfully parsed {len(instructions)} instructions!')

# Save as JSONL
with open('data/elleci_instructions.jsonl', 'w', encoding='utf-8') as f:
    for obj in instructions:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

# Stats
total_chars = sum(len(obj.get('output', '')) for obj in instructions)
print(f'Average output: {total_chars/len(instructions):.0f} chars')
print(f'Total content: {total_chars:,} chars')
print(f'Saved to data/elleci_instructions.jsonl')
