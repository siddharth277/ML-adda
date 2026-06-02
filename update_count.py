import os
import re

def count_projects():
    # This ignores system folders so it only counts your actual ML projects
    ignore_dirs = ['.git', '.github', 'venv', '__pycache__']
    
    # This detects and counts the folders
    projects = [
        d for d in os.listdir('.') 
        if os.path.isdir(d) and d not in ignore_dirs
    ]
    return len(projects)

def update_readme(count):
    with open('README.md', 'r', encoding='utf-8') as file:
        readme_content = file.read()

    # This finds your placeholder and replaces the 0 with the real count
    pattern = r'()(.*?)()'
    replacement = rf'\g<1>**{count} Projects**\g<3>'
    
    updated_readme = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)

    with open('README.md', 'w', encoding='utf-8') as file:
        file.write(updated_readme)

if __name__ == '__main__':
    project_count = count_projects()
    update_readme(project_count)
