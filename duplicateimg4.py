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

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

FRAME_STEP = 90
PHASH_THRESHOLD = 12

def hash_md5(path):
    try:
        md5_hash = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating MD5 for {path}: {e}")
        return None

def get_image_metadata(path):
    try:
        img = Image.open(path)
        stat = os.stat(path)
        return {
            "path": path,
            "size_bytes": stat.st_size,
            "resolution": img.size,
            "pixels": img.size[0] * img.size[1],
            "mode": img.mode,
            "format": img.format,
            "modified_date": stat.st_mtime,
            "created_date": stat.st_ctime
        }
    except Exception as e:
        print(f"Error reading metadata for {path}: {e}")
        return None

def get_video_metadata(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        stat = os.stat(path)
        bitrate = (stat.st_size * 8) / duration if duration > 0 else 0

        return {
            "path": path,
            "size_bytes": stat.st_size,
            "resolution": (width, height),
            "pixels": width * height,
            "duration": duration,
            "fps": fps,
            "bitrate": bitrate,
            "modified_date": stat.st_mtime,
            "created_date": stat.st_ctime
        }
    except Exception as e:
        print(f"Error reading video metadata {path}: {e}")
        return None

def choose_best_file(file1, file2, meta1, meta2, file_type="image"):
    points_1 = 0
    points_2 = 0
    reasons = []

    if meta1["size_bytes"] > meta2["size_bytes"]:
        diff_mb = (meta1["size_bytes"] - meta2["size_bytes"]) / (1024 * 1024)
        if diff_mb > 0.1:
            points_1 += 3
            reasons.append(f"File 1 is {diff_mb:.2f}MB larger")
    elif meta2["size_bytes"] > meta1["size_bytes"]:
        diff_mb = (meta2["size_bytes"] - meta1["size_bytes"]) / (1024 * 1024)
        if diff_mb > 0.1:
            points_2 += 3
            reasons.append(f"File 2 is {diff_mb:.2f}MB larger")

    if meta1["pixels"] > meta2["pixels"]:
        points_1 += 5
        reasons.append(f"File 1 has higher resolution ({meta1['resolution']})")
    elif meta2["pixels"] > meta1["pixels"]:
        points_2 += 5
        reasons.append(f"File 2 has higher resolution ({meta2['resolution']})")

    if file_type == "image":
        hierarchy = {"PNG": 3, "TIFF": 3, "BMP": 2, "WEBP": 2, "JPEG": 1, "JPG": 1, "GIF": 0}
        f1 = meta1.get("format", "").upper()
        f2 = meta2.get("format", "").upper()

        if f1 in hierarchy and f2 in hierarchy:
            if hierarchy[f1] > hierarchy[f2]:
                points_1 += 2
                reasons.append(f"File 1 has a better format ({f1} vs {f2})")
            elif hierarchy[f2] > hierarchy[f1]:
                points_2 += 2
                reasons.append(f"File 2 has a better format ({f2} vs {f1})")

    if file_type == "video":
        bitrate1 = meta1.get("bitrate", 0)
        bitrate2 = meta2.get("bitrate", 0)

        if bitrate1 > bitrate2 * 1.1:
            points_1 += 4
            reasons.append(f"File 1 has higher bitrate")
        elif bitrate2 > bitrate1 * 1.1:
            points_2 += 4
            reasons.append(f"File 2 has higher bitrate")

        fps1 = meta1.get("fps", 0)
        fps2 = meta2.get("fps", 0)

        if fps1 > fps2:
            points_1 += 1
            reasons.append(f"File 1 has higher FPS")
        elif fps2 > fps1:
            points_2 += 1
            reasons.append(f"File 2 has higher FPS")

    date1 = min(meta1.get("created_date", 9999999999), meta1.get("modified_date", 9999999999))
    date2 = min(meta2.get("created_date", 9999999999), meta2.get("modified_date", 9999999999))

    if date1 < date2:
        points_1 += 2
        reasons.append("File 1 is older (likely the original)")
    elif date2 < date1:
        points_2 += 2
        reasons.append("File 2 is older (likely the original)")

    name1 = os.path.basename(file1).lower()
    name2 = os.path.basename(file2).lower()

    copy_indicators = ["copy", "duplicate", "(1)", "(2)", "_copy", "_duplicate"]

    is_copy_1 = any(ind in name1 for ind in copy_indicators)
    is_copy_2 = any(ind in name2 for ind in copy_indicators)

    if is_copy_2 and not is_copy_1:
        points_1 += 3
        reasons.append("File 2 seems to be a copy based on its name")
    elif is_copy_1 and not is_copy_2:
        points_2 += 3
        reasons.append("File 1 seems to be a copy based on its name")

    if points_1 > points_2:
        return file1, file2, f"Keep file 1 ({points_1} vs {points_2}): {'; '.join(reasons)}"
    elif points_2 > points_1:
        return file2, file1, f"Keep file 2 ({points_2} vs {points_1}): {'; '.join(reasons)}"
    else:
        if meta1["size_bytes"] >= meta2["size_bytes"]:
            return file1, file2, "Tie – keeping file 1 (greater or equal size)"
        else:
            return file2, file1, "Tie – keeping file 2 (greater size)"

def hash_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return str(imagehash.phash(img))
    except Exception as e:
        print(f"Error computing pHash for {path}: {e}")
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
        print(f"Error: {e}")
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

        if dist <= PHASH_THRESHOLD and bit_similar > 0.65:
            similar_frames += 1

    similar_percentage = similar_frames / m

    if similar_percentage < min_similar_percentage:
        return 9999

    return sum(distances) / len(distances)

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
    for root, dirs, fs in os.walk(directory):
        for f in fs:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXT or ext in VIDEO_EXT:
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

def delete_permanently(files):
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
                base, ext = os.path.splitext(name)
                dest_path = os.path.join(destination, f"{base}_{counter}{ext}")
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
            return False, "No supported files found."

        self.update_progress("Processing files...", 20)
        data = []
        image_phashes = []
        video_hashes = []

        total = len(files)
        for idx, (path, ext) in enumerate(files):
            progress = 20 + (idx / total) * 40
            self.update_progress(f"Processing {idx+1}/{total}", progress)

            record = {"path": path, "type": "", "md5": None, "phash": None, "videohash": None}

            if ext in IMAGE_EXT:
                record["type"] = "image"
                record["md5"] = hash_md5(path)
                record["phash"] = hash_image(path)
                record["metadata"] = get_image_metadata(path)
                if record["phash"]:
                    image_phashes.append(record)

            elif ext in VIDEO_EXT:
                record["type"] = "video"
                record["md5"] = hash_md5(path)
                videohash, info = hash_video(path)
                record["videohash"] = videohash
                record["video_info"] = info
                record["metadata"] = get_video_metadata(path)

                if record["videohash"]:
                    video_hashes.append(record)
            data.append(record)

        self.update_progress("Detecting exact duplicates...", 60)
        md5_dict = {}
        md5_metadata = {}

        for item in data:
            if not item["md5"]:
                continue
            if item["md5"] in md5_dict:
                file1 = md5_dict[item["md5"]]
                file2 = item["path"]
                meta1 = md5_metadata[item["md5"]]
                meta2 = item["metadata"]

                type_item = item["type"]
                keep, remove, reason = choose_best_file(
                    file1, file2, meta1, meta2, type_item
                )

                self.exact_duplicates.append((keep, remove))
                print(f"Duplicate MD5 found: {reason}")

                if keep == file2:
                    md5_dict[item["md5"]] = file2
                    md5_metadata[item["md5"]] = meta2
            else:
                md5_dict[item["md5"]] = item["path"]
                md5_metadata[item["md5"]] = item["metadata"]

        self.update_progress("Comparing images...", 70)
        total_cmp = len(image_phashes) * (len(image_phashes) - 1) // 2
        current_cmp = 0

        for i in range(len(image_phashes)):
            for j in range(i + 1, len(image_phashes)):
                current_cmp += 1
                if current_cmp % 100 == 0:
                    progress = 70 + (current_cmp / max(total_cmp, 1)) * 10
                    self.update_progress("Comparing images...", progress)

                h1 = image_phashes[i]["phash"]
                h2 = image_phashes[j]["phash"]

                if not h1 or not h2:
                    continue

                dist = hamming_distance(h1, h2)
                if dist <= PHASH_THRESHOLD:
                    file1 = image_phashes[i]["path"]
                    file2 = image_phashes[j]["path"]
                    meta1 = image_phashes[i]["metadata"]
                    meta2 = image_phashes[j]["metadata"]

                    keep, remove, reason = choose_best_file(
                        file1, file2, meta1, meta2, "image"
                    )

                    self.image_duplicates.append((keep, remove, dist))
                    print(f"Similar image: {reason}")

        self.update_progress("Comparing videos...", 85)
        for i in range(len(video_hashes)):
            for j in range(i + 1, len(video_hashes)):

                h1 = video_hashes[i]["videohash"]
                h2 = video_hashes[j]["videohash"]

                if not h1 or not h2:
                    continue

                info1 = video_hashes[i]["video_info"]
                info2 = video_hashes[j]["video_info"]

                dur1 = info1["duration"] if info1 and info1["duration"] else None
                dur2 = info2["duration"] if info2 and info2["duration"] else None

                if info1 and info2:
                    if dur1 and dur2:
                        dur_diff = abs(dur1 - dur2) / max(dur1, dur2)
                        if dur_diff > 0.15:
                            continue

                    size_diff = abs(info1["size"] - info2["size"]) / max(info1["size"], info2["size"])
                    if size_diff > 0.30:
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

                    res_diff = abs(area1 - area2) / max(area1, area2)
                    if res_diff > 0.60:
                        continue

                dist = compare_videos(h1, h2, dur1, dur2)

                if dist <= PHASH_THRESHOLD:
                    file1 = video_hashes[i]["path"]
                    file2 = video_hashes[j]["path"]
                    meta1 = video_hashes[i]["metadata"]
                    meta2 = video_hashes[j]["metadata"]

                    keep, remove, reason = choose_best_file(
                        file1, file2, meta1, meta2, "video"
                    )

                    pair = (keep, remove)
                    if pair not in {(a, b) for a, b, _ in self.video_duplicates} and \
                       (remove, keep) not in {(a, b) for a, b, _ in self.video_duplicates}:
                        self.video_duplicates.append((keep, remove, dist))
                        print(f"Similar video: {reason}")

        self.update_progress("Analysis completed", 100)
        return True, "Analysis completed successfully."

class DuplicateFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Duplicate File Detector v2.0")
        self.root.geometry("750x700")
        self.root.resizable(False, False)

        self.analyzer = None
        self.selected_folder = None

        self.create_interface()

    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0)

        title = ttk.Label(main_frame, text="Duplicate File Detector",
                          font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        subtitle = ttk.Label(main_frame, text="Intelligent Quality-Based Prioritization",
                             font=("Arial", 9), foreground="gray")
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        ttk.Label(main_frame, text="Folder to analyze:").grid(row=2, column=0, sticky=tk.W, pady=5)

        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20))

        self.entry_folder = ttk.Entry(folder_frame, width=60)
        self.entry_folder.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(folder_frame, text="Browse", command=self.select_folder).pack(side=tk.LEFT)

        self.btn_analyze = ttk.Button(main_frame, text="Start Analysis",
                                      command=self.start_analysis, width=20)
        self.btn_analyze.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Label(main_frame, text="Progress:").grid(row=5, column=0, sticky=tk.W, pady=(20, 5))

        self.progress = ttk.Progressbar(main_frame, length=700, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=2, pady=5)

        self.label_status = ttk.Label(main_frame, text="Waiting...", foreground="gray")
        self.label_status.grid(row=7, column=0, columnspan=2, pady=5)

        ttk.Label(main_frame, text="Results:", font=("Arial", 10, "bold")).grid(
            row=8, column=0, sticky=tk.W, pady=(20, 5))

        self.text_results = tk.Text(main_frame, height=10, width=80, state=tk.DISABLED)
        self.text_results.grid(row=9, column=0, columnspan=2, pady=5)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.text_results.yview)
        scrollbar.grid(row=9, column=2, sticky=(tk.N, tk.S))
        self.text_results.config(yscrollcommand=scrollbar.set)

        action_frame = ttk.LabelFrame(main_frame, text="Duplicate Actions", padding="10")
        action_frame.grid(row=10, column=0, columnspan=2, pady=(20, 0))

        self.btn_trash = ttk.Button(action_frame, text="Move to Trash",
                                    command=self.move_to_trash, state=tk.DISABLED, width=20)
        self.btn_trash.grid(row=0, column=0, padx=5, pady=5)

        self.btn_delete = ttk.Button(action_frame, text="Delete Permanently",
                                     command=self.delete_permanently, state=tk.DISABLED, width=20)
        self.btn_delete.grid(row=0, column=1, padx=5, pady=5)

        self.btn_move = ttk.Button(action_frame, text="Move to Folder",
                                   command=self.move_to_folder, state=tk.DISABLED, width=20)
        self.btn_move.grid(row=0, column=2, padx=5, pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder to Analyze")
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
        self.disable_action_buttons()

        thread = Thread(target=self.run_analysis, daemon=True)
        thread.start()

    def run_analysis(self):
        self.analyzer = DuplicateAnalyzer(self.selected_folder, self.update_progress)
        success, msg = self.analyzer.analyze()

        if success:
            self.show_results()
            self.enable_action_buttons()
        else:
            messagebox.showerror("Error", msg)

        self.btn_analyze.config(state=tk.NORMAL)

    def show_results(self):
        self.text_results.config(state=tk.NORMAL)
        self.text_results.delete(1.0, tk.END)

        total_exact = len(self.analyzer.exact_duplicates)
        total_images = len(self.analyzer.image_duplicates)
        total_videos = len(self.analyzer.video_duplicates)

        result = "Analysis completed successfully.\n\n"
        result += f"Exact duplicates: {total_exact}\n"
        result += f"Similar images: {total_images}\n"
        result += f"Similar videos: {total_videos}\n\n"
        result += "Higher-quality originals have been preserved.\n"
        result += "Only lower-quality duplicates will be removed.\n\n"

        if total_exact > 0:
            result += "Examples of duplicates (duplicates will be removed):\n"
            for keep, remove in self.analyzer.exact_duplicates[:5]:
                result += f"  Keep: {os.path.basename(keep)}\n"
                result += f"  Remove: {os.path.basename(remove)}\n\n"

        self.text_results.insert(1.0, result)
        self.text_results.config(state=tk.DISABLED)

    def get_secondary_duplicates(self):
        duplicates = set()

        for _, remove in self.analyzer.exact_duplicates:
            duplicates.add(remove)

        for _, remove, _ in self.analyzer.image_duplicates:
            duplicates.add(remove)

        for _, remove, _ in self.analyzer.video_duplicates:
            duplicates.add(remove)

        return list(duplicates)

    def move_to_trash(self):
        if not self.confirm_action("move to trash"):
            return

        files = self.get_secondary_duplicates()
        moved, errors = move_to_trash(files)

        msg = f"{moved} duplicate files moved to the trash.\n"
        msg += "Higher-quality originals have been preserved."
        if errors:
            msg += f"\n\nErrors: {len(errors)}"

        messagebox.showinfo("Completed", msg)
        self.disable_action_buttons()

    def delete_permanently(self):
        answer = messagebox.askyesno(
            "Confirm Deletion",
            "This action will permanently delete the lower-quality duplicates.\n"
            "Higher-quality originals will be preserved.\n\n"
            "Do you want to continue?",
            icon='warning'
        )

        if not answer:
            return

        files = self.get_secondary_duplicates()
        deleted, errors = delete_permanently(files)

        msg = f"{deleted} duplicate files permanently deleted.\n"
        msg += "Higher-quality originals have been preserved."
        if errors:
            msg += f"\n\nErrors: {len(errors)}"

        messagebox.showinfo("Completed", msg)
        self.disable_action_buttons()

    def move_to_folder(self):
        destination = filedialog.askdirectory(title="Select destination folder")
        if not destination:
            return

        if not self.confirm_action(f"move to the folder:\n{destination}"):
            return

        files = self.get_secondary_duplicates()
        moved, errors = move_to_folder(files, destination)

        msg = f"{moved} duplicate files moved to:\n{destination}\n\n"
        msg += "Higher-quality originals have been preserved."
        if errors:
            msg += f"\n\nErrors: {len(errors)}"

        messagebox.showinfo("Completed", msg)
        self.disable_action_buttons()

    def confirm_action(self, action):
        total = len(self.get_secondary_duplicates())
        return messagebox.askyesno(
            "Confirm Action",
            f"Do you want to {action} {total} duplicate files?\n\n"
            f"Higher-quality originals will be preserved.\n"
            f"Only lower-quality duplicates will be removed."
        )

    def enable_action_buttons(self):
        self.btn_trash.config(state=tk.NORMAL)
        self.btn_delete.config(state=tk.NORMAL)
        self.btn_move.config(state=tk.NORMAL)

    def disable_action_buttons(self):
        self.btn_trash.config(state=tk.DISABLED)
        self.btn_delete.config(state=tk.DISABLED)
        self.btn_move.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    DuplicateFinderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

