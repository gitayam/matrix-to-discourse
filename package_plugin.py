import os
import zipfile

def zipdir(path, ziph):
    # Zip all files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

# Create a zip file
with zipfile.ZipFile('plugin.mbp', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('LICENSE')
    zipf.write('README.md')
    zipf.write('base-config.yaml')
    zipf.write('maubot.yaml')
    zipf.write('requirements.txt')
    zipdir('matrix_to_discourse', zipf)

