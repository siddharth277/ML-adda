import os
import re

def count_projects():
    """
    Counts only real ML project folders, ignoring system/config directories.
    A folder is considered a project if it contains at least one file
    (so empty folders are excluded too).
    """
    ignore_dirs = {
        '.git', '.github', 'venv', '.venv', '__pycache__',
        'node_modules', '.idea', '.vscode', 'env'
    }

    projects = []
    for d in os.listdir('.'):
        if os.path.isdir(d) and d not in ignore_dirs:
            # Only count if folder has at least one file inside
            has_files = any(
                os.path.isfile(os.path.join(d, f))
                for f in os.listdir(os.path.join(d))
            )
            if has_files:
                projects.append(d)

    print(f"[INFO] Detected {len(projects)} project(s): {projects}")
    return len(projects)

def update_readme(count):
    """
    Finds the <!-- PROJECT_COUNT --> ... <!-- END_PROJECT_COUNT --> block
    in README.md and replaces its content with the live count.
    """
    readme_path = 'README.md'

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Matches everything between the two HTML comment markers
    pattern = r'(<!-- PROJECT_COUNT -->)(.*?)(<!-- END_PROJECT_COUNT -->)'
    replacement = rf'\g<1>**{count} Projects**\g<3>'

    updated = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if updated == content:
        print("[WARN] README was not updated — markers not found or count unchanged.")
    else:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[INFO] README.md updated → {count} Projects")

if __name__ == '__main__':
    project_count = count_projects()
    update_readme(project_count)
