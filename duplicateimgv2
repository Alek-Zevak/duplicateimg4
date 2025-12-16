import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import cv2
from typing import Tuple


class DuplicateFinder:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.file_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.all_files = []
        self.file_metadata = {}
        self.parent = {}
        self.rank = {}
        self.clusters = []
        self.confirmed_duplicates = []

        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

    def get_file_size_kb(self, file_path: Path) -> int:
        return file_path.stat().st_size // 1024

    def get_image_resolution(self, file_path: Path) -> Tuple[int, int]:
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception:
            return (0, 0)

    def get_video_resolution(self, file_path: Path) -> Tuple[int, int]:
        try:
            cap = cv2.VideoCapture(str(file_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (width, height)
        except Exception:
            return (0, 0)

    def normalize_name(self, file_path: Path) -> str:
        name = file_path.stem.lower()
        name = ''.join(c for c in name if c.isalnum() or c in [' ', '-', '_'])
        return ' '.join(name.split())

    def calculate_hash(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def phase1_mapping(self):
        for file_path in self.folder_path.rglob('*'):
            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()

            if extension in self.image_extensions:
                file_type = "image"
                resolution = self.get_image_resolution(file_path)
            elif extension in self.video_extensions:
                file_type = "video"
                resolution = self.get_video_resolution(file_path)
            else:
                continue

            size_kb = self.get_file_size_kb(file_path)
            normalized_name = self.normalize_name(file_path)

            self.file_metadata[file_path] = {
                'type': file_type,
                'extension': extension,
                'size_kb': size_kb,
                'resolution': resolution,
                'normalized_name': normalized_name
            }

            self.file_map[file_type][extension][size_kb].append(file_path)
            self.all_files.append(file_path)

    def find(self, file):
        if self.parent[file] != file:
            self.parent[file] = self.find(self.parent[file])
        return self.parent[file]

    def union(self, file1, file2):
        root1 = self.find(file1)
        root2 = self.find(file2)

        if root1 == root2:
            return

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        else:
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def shallow_match(self, file1: Path, file2: Path) -> bool:
        meta1 = self.file_metadata[file1]
        meta2 = self.file_metadata[file2]

        if meta1['resolution'] == meta2['resolution'] and meta1['resolution'] != (0, 0):
            return True

        if meta1['normalized_name'] == meta2['normalized_name']:
            return True

        return False

    def phase2_block_clustering(self):
        self.parent = {file: file for file in self.all_files}
        self.rank = {file: 0 for file in self.all_files}

        for file_type in self.file_map:
            for extension in self.file_map[file_type]:
                for size_kb in self.file_map[file_type][extension]:
                    files = self.file_map[file_type][extension][size_kb]
                    if len(files) < 2:
                        continue

                    for i in range(len(files)):
                        for j in range(i + 1, len(files)):
                            if self.shallow_match(files[i], files[j]):
                                self.union(files[i], files[j])

        cluster_map = defaultdict(list)

        for file in self.all_files:
            root = self.find(file)
            cluster_map[root].append(file)

        self.clusters = [group for group in cluster_map.values() if len(group) > 1]

    def phase3_hash_verification(self):
        for cluster in self.clusters:
            hash_groups = defaultdict(list)

            for file in cluster:
                file_hash = self.calculate_hash(file)
                if file_hash:
                    hash_groups[file_hash].append(file)

            for files in hash_groups.values():
                if len(files) > 1:
                    self.confirmed_duplicates.append(files)

    def rename_duplicates(self):
        for group in self.confirmed_duplicates:
            original = group[0]
            for index, duplicate in enumerate(group[1:], start=1):
                stem = duplicate.stem
                suffix = duplicate.suffix
                parent = duplicate.parent

                new_path = parent / f"{stem}-copy{index}{suffix}"
                counter = index

                while new_path.exists():
                    counter += 1
                    new_path = parent / f"{stem}-copy{counter}{suffix}"

                duplicate.rename(new_path)

    def run(self):
        if not self.folder_path.exists():
            print("Folder does not exist")
            return

        self.phase1_mapping()

        if len(self.all_files) < 2:
            print("Not enough files")
            return

        self.phase2_block_clustering()

        if not self.clusters:
            print("No candidates found")
            return

        self.phase3_hash_verification()

        if not self.confirmed_duplicates:
            print("No exact duplicates found")
            return

        answer = input("Rename duplicate files? (y/n): ").strip().lower()
        if answer == 'y':
            self.rename_duplicates()


if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    finder = DuplicateFinder(folder)
    finder.run()
