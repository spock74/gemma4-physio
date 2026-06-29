import re
import os

os.chdir('/Users/moraes/Documents/PROJETOS/interpretability/started-june-26/zero/docs/in-tex')

with open('main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

memoir_content = content.replace(r'\documentclass[10pt,journal,onecolumn]{article}', r'\documentclass[11pt,a4paper,oneside]{memoir}')
scrbook_content = content.replace(r'\documentclass[10pt,journal,onecolumn]{article}', r'\documentclass[11pt,a4paper,oneside]{scrbook}')

def article_to_book(text, is_scrbook=False):
    if is_scrbook:
        text = re.sub(r'\\begin\{abstract\}', r'\\chapter*{Abstract}', text)
        text = re.sub(r'\\end\{abstract\}', '', text)
    else:
        # memoir supports abstract in article mode, but for book it's better to use chapter* too or just let memoir handle it.
        # Actually memoir by default doesn't support abstract unless article option is given. 
        # So let's replace it with chapter* for memoir as well, to be safe.
        text = re.sub(r'\\begin\{abstract\}', r'\\chapter*{Abstract}', text)
        text = re.sub(r'\\end\{abstract\}', '', text)
        
    text = text.replace(r'\paragraph{', r'\subsubsection*{')
    text = text.replace(r'\subsubsection{', r'\subsection{')
    text = text.replace(r'\subsection{', r'\section{')
    text = text.replace(r'\section{', r'\chapter{')
    
    return text

with open('main_memoir.tex', 'w', encoding='utf-8') as f:
    f.write(article_to_book(memoir_content, is_scrbook=False))

with open('main_scrbook.tex', 'w', encoding='utf-8') as f:
    f.write(article_to_book(scrbook_content, is_scrbook=True))

print("Conversion complete.")
