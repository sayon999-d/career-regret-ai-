import sys

def find_broken_quotes(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_html_block = False
    in_triple_sq = False
    in_triple_dq = False
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        if 'DASHBOARD_HTML = \'\'\'' in line:
            in_html_block = True
        
        if in_html_block:
            if '\'\'\'' in line and 'DASHBOARD_HTML =' not in line:
                in_html_block = False
            continue
        if '\'\'\'' in line:
            in_triple_sq = not in_triple_sq
            continue
        if '"""' in line:
            in_triple_dq = not in_triple_dq
            continue
            
        if in_triple_sq or in_triple_dq:
            continue
            
        dq_count = line.count('"') - line.count('\\"')
        sq_count = line.count("'") - line.count("\\'")
        
        if dq_count % 2 != 0:
            print(f"Potential broken DQ at line {line_num}: {line.strip()}")
        if sq_count % 2 != 0:
            print(f"Potential broken SQ at line {line_num}: {line.strip()}")

if __name__ == "__main__":
    find_broken_quotes('/Users/sayonmanna/project3/main.py')
