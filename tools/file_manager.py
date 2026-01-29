import difflib
import os
import re

class FileManager:
    def __init__(self, file_path: list):
        for p in file_path:
            if not os.path.exists(p):
                raise FileNotFoundError(f"file or directory not found: {p}")
        self.file_path = file_path

        self.all_files = []
        for p in self.file_path:
            for root, dirs, files in os.walk(p):
                for file in files:
                    if not (file.endswith('.txt') or file.endswith('.csv') or file.endswith('.jpg') or file.endswith('.json')):
                        continue
                    full_path = os.path.join(root, file)
                    self.all_files.append(full_path)

    def fetch_file(self, key: str, limit: int = 2):
        all_files = self.all_files
        pattern = re.compile(r'^(.*?)__.*$')
        name_to_paths = {}
    
        for path in all_files:
            fname = os.path.basename(path)
            match = pattern.match(fname)
            if match:
                name = match.group(1).strip()
                name_to_paths.setdefault(name, []).append(path)
    
        if not name_to_paths:
            return []
    
        closest_names = difflib.get_close_matches(key, name_to_paths.keys(), n=1, cutoff=0.1)
    
        if not closest_names:
            return []
    
        closest_name = closest_names[:limit]
        return_paths = []
        for name in closest_name:
            return_paths.extend(name_to_paths[name])
        return return_paths