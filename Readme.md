# Duplicateimg4

## Description

This script is a tool for detecting and managing duplicate files in a directory. It can detect exact duplicates, as well as similar images and videos based on the files' fingerprints (hashes). The tool provides a graphical user interface (GUI) built with Tkinter, allowing the user to easily interact with the system to perform analysis and take actions on duplicate files.

## Requirements

To run the script, the following requirements must be met:

- **Python 3.x**

Necessary libraries:
- `Pillow` (for image handling)
- `imagehash` (for generating image hashes)
- `opencv-python` (for video processing)
- `send2trash` (for moving files to the trash)
- `numpy` (for matrix operations)
- `tkinter` (for the graphical user interface)

You can install all dependencies with the following command:

```bash
pip install pillow imagehash opencv-python send2trash numpy
```
## Usage

- Open the graphical interface: Run the script, and a window with the graphical user interface (GUI) will open.
- Select a folder: Click the "Browse" button to select the directory you want to analyze for duplicate files.
- Start the analysis: Click "Start Analysis" to begin the process. The script will scan the selected folder for images and videos and generate the corresponding hashes to compare similar files.
- Review the results: Once the analysis is complete, the results will be displayed, including:
    Exact duplicates (files with the same MD5 hash).
    Similar images (compared using pHash).
    Similar videos (compared using hashes generated from multiple frames).

- Actions on duplicates: You can choose from several actions for the duplicate files:
    Move to trash.
    Permanently delete.
    Move to a specific folder.

## Program Functions
### Exact Duplicate Detection

The script calculates the MD5 hash of each file and detects exact duplicates in the directory.

### Similar Image Detection

For images, the script uses the pHash (perceptual hash) algorithm to compare the images and find similarities.

### Similar Video Detection

For videos, the script takes several frames from the video (according to a set step) and calculates the perceptual hash for each frame. The videos are compared based on their resolution, size, and duration before performing the hash comparison.

### Actions on Duplicates
**Duplicate files can be:**

- Moved to trash: Duplicate files are moved to the system's recycle bin.
- Permanently deleted: Duplicate files are irreversibly deleted.
- Moved to a folder: Duplicate files are moved to a specific directory.

## Code Structure

### Hashing Functions:
**hash_md5(path)**: Calculates the MD5 hash of a file.
**hash_image(path)**: Calculates the perceptual hash (pHash) of an image.
**hash_video(path)**: Calculates the hashes for a video by extracting frames and generating hashes for each.
**compare_videos(v1, v2, ...)**: Compares two videos using their hashes.

### File Handling Functions:
**scan_directory(directory)**: Scans a directory for image and video files.
**move_to_trash(files)**: Moves the selected files to the trash.
**permanently_delete(files)**: Permanently deletes the files.
**move_to_folder(files, destination)**: Moves the duplicate files to a specific destination.

## Graphical Interface (GUI):

Tkinter is used for the graphical interface, allowing the user to select a folder, show the progress of the analysis, and perform actions on the duplicates.

## Analysis Logic:

The class `DuplicateAnalyzer` handles the duplicate analysis, comparing both images and videos and organizing the results.

## Suggested Modifications

1. Adjust the Frame Step (FRAME_STEP)
The frame step determines how many frames are skipped between comparisons in the video analysis. This value is currently set to 90, meaning one frame is compared every 90 frames. You can adjust this value depending on the desired accuracy or the video file size:
    Higher accuracy: Reduce the FRAME_STEP value to analyze more frames.
    Better performance: Increase the FRAME_STEP value to process fewer frames and speed up the analysis.

2. Image Hash Comparison Threshold (UMBRAL_PHASH)
The UMBRAL_PHASH value sets the tolerance threshold for image comparison using pHash. If the value is too low, it might detect too many duplicates; if it's too high, it might miss important duplicates. Adjust this value based on the type of images and videos you're analyzing.

3. Adjust Video Comparison Logic
Videos are compared based on several parameters, such as duration, resolution, and size. If you want to adjust the sensitivity of video comparison, you can modify the following thresholds:
    **duration_limit**: Modify the allowed difference in video duration.
    **min_similar_percentage**: Adjust the minimum percentage of similar frames required to consider two videos as duplicates.

4. Add New File Types

Currently, the script is set to handle images and videos in certain formats. If you want to include other file types, you can add their extensions to the EXT_IMAGE and EXT_VIDEO variables. You would also need to implement an appropriate comparison algorithm for those file types.
