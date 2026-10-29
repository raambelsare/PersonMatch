# Person Match

Compare two face photos from the command line to see if they are the same person.
It uses ArcFace by default, which is more reliable than the previous model when
the same person has changed facial hair, age, lighting, or camera angle.

## Setup

```sh
pip install -r requirements.txt
```

The first comparison downloads ArcFace's recognition weights once (about
137 MB) into DeepFace's local model cache. Later comparisons use that local
copy and do not need another download.

## How to Copy a File Path

- **Mac**: Right-click the file, hold the `Option` (⌥) key, and click **Copy "..." as Pathname** (or press `Option + Command + C`). You can also drag and drop the file directly into the Terminal.
- **Windows**: Right-click the file and click **Copy as path** (or press `Ctrl + Shift + C`). You can also drag and drop the file directly into Command Prompt or PowerShell.

## Usage

### Option 1: Pass paths directly
```sh
python3 person_matcher.py /path/to/image1.jpg /path/to/image2.jpg
```

### Option 2: Interactive prompt
Run without arguments and paste or drag-and-drop each photo when prompted:
```sh
python3 person_matcher.py
```

> **Note for Mac users**: If you see a permission error accessing photos in `Downloads` or `Desktop`, move the photos into this project folder or grant Terminal access in **System Settings → Privacy & Security → Files and Folders**.

For details on how the face recognition pipeline and mathematics work under the hood, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
