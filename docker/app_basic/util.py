import sys
import os
import subprocess
from typing import Any


def download_spanish_model(name_model:str)->Any:

    """Downland Models"""

    if not name_model in ["es_core_news_sm","es_core_news_md","es_core_news_lg"]:
        raise ValueError("Wrong name of model")
     

    command=['python','-m','spacy','download','{0}'.format(name_model)]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, shell=False,
           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

    proc.stdin.close()
    original_output = proc.stdout.read()
    proc.wait()

    if proc.returncode ==1:
        raise ValueError(original_output.decode("utf-8").strip())

#https://github.com/explosion/spaCy/issues/4592
def list_available_models():
    """ Scan PYTHONPATH to find spacy models """
    models = []
    # For each directory in PYTHONPATH
    paths = [p for p in sys.path if os.path.isdir(p)]
    for site_package_dir in paths:
        # For each module
        modules = [os.path.join(site_package_dir, m) for m in os.listdir(site_package_dir)]
        modules = [m for m in modules if os.path.isdir(m)]
        for module_dir in modules:
            if 'meta.json' in os.listdir(module_dir):
                # Ensure the package we're in is a spacy model
                meta_path = os.path.join(module_dir, 'meta.json')
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('parent_package', '') == 'spacy':
                    models.append(module_dir)
    return models


def get_spacy_models(filt=None):
    """ Return a dictionnary {model_path: link_name or None} """
    linked = list_linked_models()
    models = list_available_models()
    # Make `models` as `linked` (a dictionnary {model_path: None})
    models = {m: None for m in models}
    # Replace None by `link_name` if exists
    models.update(linked)
    if filt:
        # Hack to filter module of a specific lang generally spacy's
        # module name begins with alpha2 language code
        models = {m: l for m, l in models.items() if os.path.basename(m)[:2] == filt}
    return models