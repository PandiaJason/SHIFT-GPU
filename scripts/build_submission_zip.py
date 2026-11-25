import zipfile
import os

zip_name = 'SHIFT_GPU_Elsevier_Submission_Package.zip'
print(f'Creating {zip_name}...')

# Files and folders to include
items_to_zip = [
    'paper',
    'src',
    'scripts',
    'experiments',
    'README.md',
    'main.tex',
    'main.pdf',
    'references.bib',
    'SHIFT_GPU_Full_Paper.pdf',
    'cover_letter.tex',
    'cover_letter.pdf',
    'highlights.tex'
]

exclude_exts = {'.pyc', '.aux', '.log', '.out', '.toc', '.fls', '.fdb_latexmk', '.synctex.gz'}
exclude_names = {'.DS_Store', '__pycache__'}

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    for item in items_to_zip:
        if os.path.isfile(item):
            zf.write(item, item)
            print(f'Added file: {item}')
        elif os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d not in exclude_names]
                for file in files:
                    if file in exclude_names or any(file.endswith(ext) for ext in exclude_exts):
                        continue
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, '.')
                    zf.write(full_path, arcname)
                    # print(f'Added: {arcname}')

print(f'Successfully built {zip_name}')
