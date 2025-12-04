import os
import hashlib
import shutil
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
import send2trash
import numpy as np

EXT_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
EXT_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

FRAME_STEP = 90
UMBRAL_PHASH = 12

def hash_md5(path):
    try:
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating MD5 of {path}: {e}")
        return None

def hash_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception as e:
        print(f"Error calculating pHash of {path}: {e}")
        return None

def hash_video(path, frame_step=FRAME_STEP):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else None
        size_bytes = os.path.getsize(path)

        hashes = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hashes.append(hash_frame_multi(frame))

            frame_id += 1

        cap.release()

        info = {
            "duration": duration,
            "resolution": (width, height),
            "aspect": width / height if height > 0 else None,
            "fps": fps,
            "size": size_bytes
        }

        return hashes if hashes else None, info
    except Exception as e:
        print(f"Error processing video {path}: {e}")
        return None, None

def hamming_distance(h1, h2):
    try:
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
    except Exception as e:
        print(f"Error computing Hamming distance: {e}")
        return 9999

def compare_videos(v1, v2, dur1, dur2,
                   duration_limit=0.10,
                   min_similar_percentage=0.60):

    if not v1 or not v2:
        return 9999

    m = min(len(v1), len(v2))
    if m == 0:
        return 9999

    distances = []
    similar_frames = 0

    for i in range(m):
        A = v1[i]
        B = v2[i]

        dp = A["phash"] - B["phash"]
        da = A["ahash"] - B["ahash"]
        dd = A["dhash"] - B["dhash"]
        de = A["edge"] - B["edge"]

        dist = (dp + da + dd + de) / 4
        distances.append(dist)

        frameA = np.array(A["phash"].hash, dtype=np.uint8)
        frameB = np.array(B["phash"].hash, dtype=np.uint8)
        bit_similar = (frameA == frameB).mean()

        if dist <= UMBRAL_PHASH and bit_similar > 0.65:
            similar_frames += 1

    similar_percentage = similar_frames / m

    if similar_percentage < min_similar_percentage:
        return 9999

    return sum(distances) / len(distances)

def video_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frames / fps
    return None

def hash_frame_multi(frame):
    img = Image.fromarray(frame)

    ph = imagehash.phash(img)
    ah = imagehash.average_hash(img)
    dh = imagehash.dhash(img)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edges_img = Image.fromarray(edges)
    edge_hash = imagehash.phash(edges_img)

    return {
        "phash": ph,
        "ahash": ah,
        "dhash": dh,
        "edge": edge_hash
    }

def scan_directory(directory):
    files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in EXT_IMAGE or ext in EXT_VIDEO:
                files.append((path, ext))
    return files

def move_to_trash(files):
    moved = 0
    errors = []
    for file in files:
        try:
            send2trash.send2trash(file)
            moved += 1
        except Exception as e:
            errors.append(f"{file}: {e}")
    return moved, errors

def permanently_delete(files):
    deleted = 0
    errors = []
    for file in files:
        try:
            os.remove(file)
            deleted += 1
        except Exception as e:
            errors.append(f"{file}: {e}")
    return deleted, errors

def move_to_folder(files, destination):
    Path(destination).mkdir(parents=True, exist_ok=True)
    moved = 0
    errors = []

    for file in files:
        try:
            name = os.path.basename(file)
            dest_path = os.path.join(destination, name)

            counter = 1
            while os.path.exists(dest_path):
                name_base, ext = os.path.splitext(name)
                dest_path = os.path.join(destination, f"{name_base}_{counter}{ext}")
                counter += 1

            shutil.move(file, dest_path)
            moved += 1
        except Exception as e:
            errors.append(f"{file}: {e}")

    return moved, errors

class DuplicateAnalyzer:
    def __init__(self, folder, progress_callback=None):
        self.folder = folder
        self.progress_callback = progress_callback
        self.exact_duplicates = []
        self.image_duplicates = []
        self.video_duplicates = []

    def update_progress(self, message, percentage):
        if self.progress_callback:
            self.progress_callback(message, percentage)

    def analyze(self):
        self.update_progress("Scanning files...", 10)
        files = scan_directory(self.folder)

        if not files:
            return False, "No compatible files found."

        self.update_progress("Processing files...", 20)
        data = []
        image_phash = []
        video_hash = []

        total = len(files)
        for idx, (path, ext) in enumerate(files):
            progress = 20 + (idx / total) * 40
            self.update_progress(f"Processing {idx+1}/{total}", progress)

            record = {"path": path, "type": "", "md5": None, "phash": None, "videohash": None}

            if ext in EXT_IMAGE:
                record["type"] = "image"
                record["md5"] = hash_md5(path)
                record["phash"] = hash_image(path)
                if record["phash"]:
                    image_phash.append(record)

            elif ext in EXT_VIDEO:
                record["type"] = "video"
                record["md5"] = hash_md5(path)
                videohash, info = hash_video(path)
                record["videohash"] = videohash
                record["video_info"] = info

                if record["videohash"]:
                    video_hash.append(record)
            data.append(record)

        self.update_progress("Detecting exact duplicates...", 60)
        md5_dict = {}
        for item in data:
            if not item["md5"]:
                continue
            if item["md5"] in md5_dict:
                self.exact_duplicates.append((md5_dict[item["md5"]], item["path"]))
            else:
                md5_dict[item["md5"]] = item["path"]

        self.update_progress("Comparing images...", 70)
        total_comparisons = len(image_phash) * (len(image_phash) - 1) // 2
        current_comparison = 0

        for i in range(len(image_phash)):
            for j in range(i + 1, len(image_phash)):
                current_comparison += 1
                if current_comparison % 100 == 0:
                    progress = 70 + (current_comparison / max(total_comparisons, 1)) * 10
                    self.update_progress("Comparing images...", progress)

                h1 = image_phash[i]["phash"]
                h2 = image_phash[j]["phash"]

                if not h1 or not h2:
                    continue

                dist = hamming_distance(h1, h2)
                if dist <= UMBRAL_PHASH:
                    self.image_duplicates.append((image_phash[i]["path"],
                                                  image_phash[j]["path"],
                                                  dist))

        self.update_progress("Comparing videos...", 85)
        for i in range(len(video_hash)):
            for j in range(i + 1, len(video_hash)):

                h1 = video_hash[i]["videohash"]
                h2 = video_hash[j]["videohash"]

                if not h1 or not h2:
                    continue

                info1 = video_hash[i]["video_info"]
                info2 = video_hash[j]["video_info"]

                dur1 = info1["duration"] if info1 and info1["duration"] else None
                dur2 = info2["duration"] if info2 and info2["duration"] else None

                if info1 and info2:

                    if dur1 and dur2:
                        dif_dur = abs(dur1 - dur2) / max(dur1, dur2)
                        if dif_dur > 0.15:
                            continue

                    dif_size = abs(info1["size"] - info2["size"]) / max(info1["size"], info2["size"])
                    if dif_size > 0.30:
                        continue

                    w1, h1_res = info1["resolution"]
                    w2, h2_res = info2["resolution"]

                    asp1 = info1["aspect"]
                    asp2 = info2["aspect"]
                    if asp1 and asp2:
                        if abs(asp1 - asp2) > 0.15:
                            continue
                    
                    area1 = w1 * h1_res
                    area2 = w2 * h2_res

                    dif_res = abs(area1 - area2) / max(area1, area2)
                    if dif_res > 0.60:
                        continue

                dist = compare_videos(h1, h2, dur1, dur2)

                if dist <= UMBRAL_PHASH:
                    pair = (video_hash[i]["path"], video_hash[j]["path"])

                    if pair not in {(a, b) for a, b, _ in self.video_duplicates} and \
                        (pair[1], pair[0]) not in {(a, b) for a, b, _ in self.video_duplicates}:

                        self.video_duplicates.append(
                            (pair[0], pair[1], dist)
                        )
        self.update_progress("Analysis completed", 100)
        return True, "Analysis completed successfully."

class GUIApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Duplicate File Detector")
        self.root.geometry("675x650")
        self.root.resizable(False, False)

        self.analyzer = None
        self.selected_folder = None

        self.create_interface()

    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0)

        title = ttk.Label(main_frame, text="Duplicate File Detector",
                           font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        ttk.Label(main_frame, text="Folder to analyze:").grid(row=1, column=0, sticky=tk.W, pady=5)

        frame_folder = ttk.Frame(main_frame)
        frame_folder.grid(row=2, column=0, columnspan=2, pady=(0, 20))

        self.entry_folder = ttk.Entry(frame_folder, width=50)
        self.entry_folder.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(frame_folder, text="Browse", command=self.select_folder).pack(side=tk.LEFT)

        self.btn_analyze = ttk.Button(main_frame, text="Start Analysis",
                                       command=self.start_analysis, width=20)
        self.btn_analyze.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Label(main_frame, text="Progress:").grid(row=4, column=0, sticky=tk.W, pady=(20, 5))

        self.progress = ttk.Progressbar(main_frame, length=600, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=2, pady=5)

        self.label_status = ttk.Label(main_frame, text="Waiting...", foreground="gray")
        self.label_status.grid(row=6, column=0, columnspan=2, pady=5)

        ttk.Label(main_frame, text="Results:", font=("Arial", 10, "bold")).grid(
            row=7, column=0, sticky=tk.W, pady=(20, 5))

        self.text_results = tk.Text(main_frame, height=8, width=70, state=tk.DISABLED)
        self.text_results.grid(row=8, column=0, columnspan=2, pady=5)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.text_results.yview)
        scrollbar.grid(row=8, column=2, sticky=(tk.N, tk.S))
        self.text_results.config(yscrollcommand=scrollbar.set)

        frame_actions = ttk.LabelFrame(main_frame, text="Actions on duplicates", padding="10")
        frame_actions.grid(row=9, column=0, columnspan=2, pady=(20, 0))

        self.btn_trash = ttk.Button(frame_actions, text="Move to Trash",
                                       command=self.move_to_trash, state=tk.DISABLED, width=20)
        self.btn_trash.grid(row=0, column=0, padx=5, pady=5)

        self.btn_delete = ttk.Button(frame_actions, text="Delete Permanently",
                                       command=self.delete_permanently, state=tk.DISABLED, width=20)
        self.btn_delete.grid(row=0, column=1, padx=5, pady=5)

        self.btn_move = ttk.Button(frame_actions, text="Move to Folder",
                                    command=self.move_to_folder, state=tk.DISABLED, width=20)
        self.btn_move.grid(row=0, column=2, padx=5, pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder to analyze")
        if folder:
            self.selected_folder = folder
            self.entry_folder.delete(0, tk.END)
            self.entry_folder.insert(0, folder)

    def update_progress(self, message, percentage):
        self.progress['value'] = percentage
        self.label_status.config(text=message)
        self.root.update_idletasks()

    def start_analysis(self):
        if not self.selected_folder:
            messagebox.showwarning("Warning", "You must select a folder.")
            return

        if not os.path.exists(self.selected_folder):
            messagebox.showerror("Error", "The selected folder does not exist.")
            return

        self.btn_analyze.config(state=tk.DISABLED)
        self.deactivate_action_buttons()

        thread = Thread(target=self.run_analysis, daemon=True)
        thread.start()

    def run_analysis(self):
        self.analyzer = DuplicateAnalyzer(self.selected_folder, self.update_progress)
        success, message = self.analyzer.analyze()

        if success:
            self.show_results()
            self.activate_action_buttons()
        else:
            messagebox.showerror("Error", message)

        self.btn_analyze.config(state=tk.NORMAL)

    def show_results(self):
        self.text_results.config(state=tk.NORMAL)
        self.text_results.delete(1.0, tk.END)

        total_exact = len(self.analyzer.exact_duplicates)
        total_images = len(self.analyzer.image_duplicates)
        total_videos = len(self.analyzer.video_duplicates)

        result = "Analysis completed\n\n"
        result += f"Exact duplicates: {total_exact}\n"
        result += f"Similar images: {total_images}\n"
        result += f"Similar videos: {total_videos}\n\n"

        if total_exact > 0:
            result += "Examples of exact duplicates:\n"
            for a, b in self.analyzer.exact_duplicates[:3]:
                result += f"  - {os.path.basename(b)}\n"

        self.text_results.insert(1.0, result)
        self.text_results.config(state=tk.DISABLED)

    def get_secondary_duplicates(self):
        duplicates = set()

        for _, b in self.analyzer.exact_duplicates:
            duplicates.add(b)

        for _, b, _ in self.analyzer.image_duplicates:
            duplicates.add(b)

        for _, b, _ in self.analyzer.video_duplicates:
            duplicates.add(b)

        return list(duplicates)

    def move_to_trash(self):
        if not self.confirm_action("move to the trash"):
            return

        files = self.get_secondary_duplicates()
        moved, errors = move_to_trash(files)

        message = f"{moved} files moved to trash."
        if errors:
            message += f"\nErrors: {len(errors)}"

        messagebox.showinfo("Completed", message)
        self.deactivate_action_buttons()

    def delete_permanently(self):
        response = messagebox.askyesno(
            "Confirm deletion",
            "This action will permanently delete the files.\nDo you want to continue?",
            icon='warning'
        )

        if not response:
            return

        files = self.get_secondary_duplicates()
        deleted, errors = permanently_delete(files)

        message = f"{deleted} files permanently deleted."
        if errors:
            message += f"\nErrors: {len(errors)}"

        messagebox.showinfo("Completed", message)
        self.deactivate_action_buttons()

    def move_to_folder(self):
        destination_folder = filedialog.askdirectory(title="Select destination folder")
        if not destination_folder:
            return

        if not self.confirm_action(f"move to folder:\n{destination_folder}"):
            return

        files = self.get_secondary_duplicates()
        moved, errors = move_to_folder(files, destination_folder)

        message = f"{moved} files moved to:\n{destination_folder}"
        if errors:
            message += f"\nErrors: {len(errors)}"

        messagebox.showinfo("Completed", message)
        self.deactivate_action_buttons()

    def confirm_action(self, action):
        total = len(self.get_secondary_duplicates())
        return messagebox.askyesno(
            "Confirm action",
            f"Do you want to {action} {total} duplicate files?"
        )

    def activate_action_buttons(self):
        self.btn_trash.config(state=tk.NORMAL)
        self.btn_delete.config(state=tk.NORMAL)
        self.btn_move.config(state=tk.NORMAL)

    def deactivate_action_buttons(self):
        self.btn_trash.config(state=tk.DISABLED)
        self.btn_delete.config(state=tk.DISABLED)
        self.btn_move.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    GUIApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()