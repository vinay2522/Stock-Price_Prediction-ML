import os
import re

templates_dir = r'd:\projects\FSD\stock_market_ml\app\templates'

# Patterns to remove
patterns_to_remove = [
    r'<link href="\{% static \'lib/owlcarousel/assets/owl\.carousel\.min\.css\' %\}" rel="stylesheet">',
    r'<link href="\{% static \'lib/tempusdominus/css/tempusdominus-bootstrap-4\.min\.css\' %\}" rel="stylesheet" />',
    r'<script src="\{% static \'lib/chart/chart\.min\.js\' %\}"></script>',
    r'<script src="\{% static \'lib/easing/easing\.min\.js\' %\}"></script>',
    r'<script src="\{% static \'lib/waypoints/waypoints\.min\.js\' %\}"></script>',
    r'<script src="\{% static \'lib/owlcarousel/owl\.carousel\.min\.js\' %\}"></script>',
    r'<script src="\{% static \'lib/tempusdominus/js/moment\.min\.js\' %\}"></script>',
    r'<script src="\{% static \'lib/tempusdominus/js/moment-timezone\.min\.js\' %\}"></script>',
    r'<script src="\{% static \'lib/tempusdominus/js/tempusdominus-bootstrap-4\.min\.js\' %\}"></script>',
]

for filename in os.listdir(templates_dir):
    if filename.endswith('.html'):
        filepath = os.path.join(templates_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content)
        
        # Remove empty lines that were left behind
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Fixed: {filename}')
        else:
            print(f'No changes: {filename}')

print('\nAll templates processed!')
