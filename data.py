from pathlib import Path
from PIL import Image, ImageOps, ImageTk
from pillow_heif import register_heif_opener
import shutil
import csv
import math
import queue
import threading
import tkinter as tk
import random

register_heif_opener()

folder = Path("data")
L = Image.Resampling.LANCZOS


X = []
y = []

rows = []
try:
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            features = row[4:]
            # charlotte, ivy, leah
            # one hot encoded
            label = [int(row[0]), int(row[1]), int(row[2])]
            new_row = []
            new_row.extend(label)
            new_row.append(row[3])
            new_row.extend(features)
            rows.append(new_row)
except:
    print("data.csv read error")

print(f"{len(rows)} prior rows loaded")

def show(imgparam, large=False):
    img = imgparam.copy()
    width = math.floor(shutil.get_terminal_size().columns * (1 if large else 0.4))
    img = img.convert("RGB")
    height = max(2, int(img.height * width / img.width) // 2 * 2)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    px = img.load()

    lines = []
    for y in range(0, height, 2):
        row = []
        for x in range(width):
            r1, g1, b1 = px[x, y]
            r2, g2, b2 = px[x, y + 1]
            row.append(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▀")
        lines.append("".join(row) + "\x1b[0m")
    print("\n".join(lines))

class Viewer:
    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.root = tk.Tk()
        self.root.title("Image viewer")
        self.root.configure(bg="black")
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.root.after(50, self._poll)
        self.root.mainloop()

    def _poll(self):
        try:
            while True:
                img, large, title = self.q.get_nowait()
                if large:
                    max_size = (int(self.root.winfo_screenwidth() * 0.9), int(self.root.winfo_screenheight() * 0.85))
                else:
                    max_size = (700, 700)
                img.thumbnail(max_size, L)
                photo = ImageTk.PhotoImage(img)
                self.label.configure(image=photo)
                self.label.image = photo
                self.root.title(title)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def show(self, img, large=False, title="Image viewer"):
        self.q.put((img.copy(), large, title))

viewer = Viewer()

def find_row(file_name):
    for r in rows:
        if r[3] == file_name:
            return r
    return None

examples = []

# n x n x 3
size = (224, 224)
print(f"TOTAL_PIXELS = {size[0] * size[1]}")

def pixels_to_gray(pixels):
    i = 0
    points = []
    while i < len(pixels):
        r = pixels[i]
        g = pixels[i + 1]
        b = pixels[i + 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        points.append(gray)
        i += 3
    return points
    
files = list(folder.iterdir())
i = -1
large = False
while i < len(files) - 1:
    i += 1
    if i >= len(files):
        break
    f = files[i]

    img = Image.open(f).convert("RGB")
    resized = ImageOps.pad(img, size, L, color=(124, 116, 104))
    pixels = list(resized.getdata()) # (r, g, b)[]
    row = find_row(f.name)
    if row is not None:
        vec = row[:4]
        
        vec.extend(pixels)
        examples.append(vec)
        print(f"filled example {i} from resizing row in data.csv")
        continue
    continue
    viewer.show(img, large, f"{i + 1}/{len(files)} {f.name}")
    large = False
    inp = input(f"Label picture {i + 1}/{len(files)} {f.name} : xyz (each 0 or 1)\nCurrent examples: {len(examples)}\nLabel: ")
    if inp == "back":
        i -= 2
        examples.pop()
        continue
    if inp == "i":
        i -= 1
        large = True
        continue
    if inp == "s":
        continue

    if len(inp) != 3:
        i -= 1
        print(f"invalid input {inp}")
        continue
    charlotte = 1 if inp[0] == "1" else 0
    ivy = 1 if inp[1] == "1" else 0
    leah = 1 if inp[2] == "1" else 0
    
    vec = [charlotte, ivy, leah, f.name]
    vec.extend(pixels)
    examples.append(vec)

def split(data, test_ratio = 0.2):
    indexes = set()
    test = []
    train = []
    test_size = math.floor(len(data) * test_ratio)
    print("test size", test_size)
    print("# examples", len(data))

    while len(test) < test_size:
        idx = random.randint(0, len(data) - 1)
        if idx in indexes:
            continue

        indexes.add(idx)
        test.append(data[idx])

    for i in range(len(data)):
        if i not in indexes:
            train.append(data[i])

    return train, test

def write(data):
    train, test = split(data)

    with open("data.csv", "w", newline="") as fh:
        print("Writing to data.csv")
        writer = csv.writer(fh)
        header = ["charlotte", "ivy", "leah", "file name"]
        header.extend([f"px{i + 1}" for i in range(size[0] * size[1])])
        writer.writerow(header)
        for ex in data:
            writer.writerow(ex)
        print(f"Wrote {len(data)} to data.csv")

    with open("train.csv", "w", newline="") as fh:
        print("Writing to train.csv")
        writer = csv.writer(fh)
        header = ["charlotte", "ivy", "leah", "file name"]
        header.extend([f"px{i + 1}" for i in range(size[0] * size[1])])
        writer.writerow(header)
        for ex in train:
            writer.writerow(ex)
        print(f"Wrote {len(train)} to train.csv")

    with open("test.csv", "w", newline="") as fh:
        print("Writing to test.csv")
        writer = csv.writer(fh)
        header = ["charlotte", "ivy", "leah", "file name"]
        header.extend([f"px{i + 1}" for i in range(size[0] * size[1])])
        writer.writerow(header)
        for ex in test:
            writer.writerow(ex)
        print(f"Wrote {len(test)} to test.csv")

write(examples)