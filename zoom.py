from ultralytics import YOLO
import os
import cv2
import shutil
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
# model_path = r"E:\Training_toy_data\Outclass Model\apple_scratch_v1777541618\weights\best.pt"
model_path = r"D:\5k Yolo weights\weights\best.pt"

input_folder = r"D:\CLIP\separated"

single_image = None

output_root = r"E:\Training_toy_data\Main Dataset\Main_Data_training"
# output_root = r"C:\Users\Virus\Desktop\New folder (7)"

output_images = os.path.join(output_root, "images")
output_labels = os.path.join(output_root, "labels")

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# LOAD IMAGES
# -----------------------------
img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

images = []

if single_image:
    images = [single_image]

else:

    for f in os.listdir(input_folder):

        if f.lower().endswith(img_ext):

            base = os.path.splitext(f)[0]

            label_path = os.path.join(output_labels, base + ".txt")

            # skip already processed
            if os.path.exists(label_path):
                continue

            images.append(os.path.join(input_folder, f))

print(f"Total images remaining: {len(images)}")

# -----------------------------
# GLOBALS
# -----------------------------
boxes = []

selected_idx = -1
mode = None
corner = None

start_x = 0
start_y = 0

CORNER_SIZE = 12

# -----------------------------
# ZOOM VARIABLES
# -----------------------------
zoom = 1.0

zoom_mode = False
zoom_rect = None

view_x1 = 0
view_y1 = 0
view_x2 = 0
view_y2 = 0

# -----------------------------
# DELETE REGION VARIABLES
# -----------------------------
delete_region_mode = False
delete_rect = None

# -----------------------------
# LOAD LABELS
# -----------------------------
def load_labels(label_path, w, h):

    if not os.path.exists(label_path):
        return None

    loaded = []

    with open(label_path, "r") as f:

        for line in f:

            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            loaded.append((x1, y1, x2, y2))

    return loaded

# -----------------------------
# CORNER DETECTION
# -----------------------------
def get_corner(x, y, box):

    x1, y1, x2, y2 = box

    if abs(x - x1) < CORNER_SIZE and abs(y - y1) < CORNER_SIZE:
        return 'tl'

    if abs(x - x2) < CORNER_SIZE and abs(y - y1) < CORNER_SIZE:
        return 'tr'

    if abs(x - x1) < CORNER_SIZE and abs(y - y2) < CORNER_SIZE:
        return 'bl'

    if abs(x - x2) < CORNER_SIZE and abs(y - y2) < CORNER_SIZE:
        return 'br'

    return None

# -----------------------------
# SCREEN -> IMAGE COORDS
# -----------------------------
def screen_to_image(x, y):

    global view_x1, view_y1, zoom

    ix = int((x / zoom) + view_x1)
    iy = int((y / zoom) + view_y1)

    return ix, iy

# -----------------------------
# MOUSE CALLBACK
# -----------------------------
def mouse_callback(event, x, y, flags, param):

    global selected_idx
    global mode
    global corner
    global boxes
    global start_x, start_y

    global zoom_mode
    global zoom_rect
    global zoom

    global delete_region_mode
    global delete_rect

    global view_x1
    global view_y1
    global view_x2
    global view_y2

    ix, iy = screen_to_image(x, y)

    # -----------------------------------
    # ZOOM SELECTION MODE
    # -----------------------------------
    if zoom_mode:

        if event == cv2.EVENT_LBUTTONDOWN:

            start_x, start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:

            if flags & cv2.EVENT_FLAG_LBUTTON:

                zoom_rect = (start_x, start_y, x, y)

        elif event == cv2.EVENT_LBUTTONUP:

            x1 = min(start_x, x)
            y1 = min(start_y, y)

            x2 = max(start_x, x)
            y2 = max(start_y, y)

            if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:

                ix1, iy1 = screen_to_image(x1, y1)
                ix2, iy2 = screen_to_image(x2, y2)

                view_x1 = ix1
                view_y1 = iy1

                view_x2 = ix2
                view_y2 = iy2

                roi_w = ix2 - ix1
                roi_h = iy2 - iy1

                win_w = 1400
                win_h = 900

                zoom_x = win_w / roi_w
                zoom_y = win_h / roi_h

                zoom = min(zoom_x, zoom_y)

            zoom_mode = False
            zoom_rect = None

        return

    # -----------------------------------
    # DELETE REGION MODE
    # -----------------------------------
    if delete_region_mode:

        if event == cv2.EVENT_LBUTTONDOWN:

            start_x, start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:

            if flags & cv2.EVENT_FLAG_LBUTTON:

                delete_rect = (start_x, start_y, x, y)

        elif event == cv2.EVENT_LBUTTONUP:

            x1 = min(start_x, x)
            y1 = min(start_y, y)

            x2 = max(start_x, x)
            y2 = max(start_y, y)

            ix1, iy1 = screen_to_image(x1, y1)
            ix2, iy2 = screen_to_image(x2, y2)

            new_boxes = []

            for box in boxes:

                bx1, by1, bx2, by2 = box

                # BOX CENTER
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2

                # KEEP boxes outside region
                if not (ix1 <= cx <= ix2 and iy1 <= cy <= iy2):

                    new_boxes.append(box)

            boxes[:] = new_boxes

            print("🗑 Deleted boxes in selected region")

            delete_region_mode = False
            delete_rect = None

        return

    # -----------------------------------
    # NORMAL ANNOTATION
    # -----------------------------------
    if event == cv2.EVENT_LBUTTONDOWN:

        for i, box in enumerate(boxes):

            c = get_corner(ix, iy, box)

            if c:

                selected_idx = i
                mode = 'resize'
                corner = c

                return

            x1, y1, x2, y2 = box

            if x1 < ix < x2 and y1 < iy < y2:

                selected_idx = i
                mode = 'move'

                start_x = ix
                start_y = iy

                return

        mode = 'draw'

        start_x = ix
        start_y = iy

    elif event == cv2.EVENT_MOUSEMOVE:

        # MOVE
        if selected_idx != -1 and mode == 'move':

            dx = ix - start_x
            dy = iy - start_y

            x1, y1, x2, y2 = boxes[selected_idx]

            boxes[selected_idx] = (
                x1 + dx,
                y1 + dy,
                x2 + dx,
                y2 + dy
            )

            start_x = ix
            start_y = iy

        # RESIZE
        elif selected_idx != -1 and mode == 'resize':

            x1, y1, x2, y2 = boxes[selected_idx]

            if corner == 'tl':
                x1, y1 = ix, iy

            elif corner == 'tr':
                x2, y1 = ix, iy

            elif corner == 'bl':
                x1, y2 = ix, iy

            elif corner == 'br':
                x2, y2 = ix, iy

            boxes[selected_idx] = (x1, y1, x2, y2)

    elif event == cv2.EVENT_LBUTTONUP:

        # DRAW
        if mode == 'draw':

            x1, y1 = start_x, start_y
            x2, y2 = ix, iy

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:

                boxes.append((
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2)
                ))

        mode = None

# -----------------------------
# WINDOW
# -----------------------------
cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("viewer", mouse_callback)

# -----------------------------
# MAIN LOOP
# -----------------------------
i = 0

while i < len(images):

    img_path = images[i]

    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]

    original_img = cv2.imread(img_path)

    if original_img is None:
        i += 1
        continue

    h, w = original_img.shape[:2]

    label_path = os.path.join(output_labels, base_name + ".txt")

    # -----------------------------
    # LOAD OR PREDICT
    # -----------------------------
    boxes = load_labels(label_path, w, h)

    if boxes is None:

        results = model.predict(
            source=img_path,
            conf=0.15,
            iou=0.7,
            verbose=False
        )

        r = results[0]

        boxes = []

        if r.boxes is not None:

            for box in r.boxes:

                boxes.append(tuple(map(int, box.xyxy[0])))

    selected_idx = -1

    # reset zoom
    zoom = 1.0

    view_x1 = 0
    view_y1 = 0
    view_x2 = 0
    view_y2 = 0

    while True:

        img = original_img.copy()

        # -----------------------------
        # DRAW BOXES
        # -----------------------------
        for j, (x1, y1, x2, y2) in enumerate(boxes):

            color = (0, 255, 0)

            if j == selected_idx:
                color = (0, 0, 255)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            for (cx, cy) in [
                (x1, y1),
                (x2, y1),
                (x1, y2),
                (x2, y2)
            ]:

                cv2.circle(
                    img,
                    (cx, cy),
                    4,
                    (255, 0, 0),
                    -1
                )

        # -----------------------------
        # APPLY VIEWPORT CROP
        # -----------------------------
        if view_x2 > view_x1 and view_y2 > view_y1:

            cropped = img[
                view_y1:view_y2,
                view_x1:view_x2
            ]

        else:
            cropped = img

        # -----------------------------
        # RESIZE DISPLAY
        # -----------------------------
        display = cv2.resize(
            cropped,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_LINEAR
        )

        # -----------------------------
        # DRAW ROI RECTANGLE
        # -----------------------------
        if zoom_rect is not None:

            zx1, zy1, zx2, zy2 = zoom_rect

            cv2.rectangle(
                display,
                (zx1, zy1),
                (zx2, zy2),
                (0, 255, 255),
                2
            )

        # -----------------------------
        # DRAW DELETE REGION
        # -----------------------------
        if delete_rect is not None:

            dx1, dy1, dx2, dy2 = delete_rect

            cv2.rectangle(
                display,
                (dx1, dy1),
                (dx2, dy2),
                (0, 0, 255),
                2
            )

        # -----------------------------
        # INFO TEXT
        # -----------------------------
        cv2.putText(
            display,
            f"{i+1}/{len(images)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "z=Zoom Area | 0=Reset View",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "a=Prev d=Next y=Save x=Delete c=DeleteRegion",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("viewer", display)

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------
        # DELETE SELECTED BOX
        # -----------------------------
        if key == ord('x'):

            if selected_idx != -1:

                boxes.pop(selected_idx)
                selected_idx = -1

        # -----------------------------
        # DELETE REGION
        # -----------------------------
        elif key == ord('c'):

            delete_region_mode = True

            print("🗑 Drag mouse to select delete region")

        # -----------------------------
        # SELECT NEXT BOX
        # -----------------------------
        elif key == 9:

            if len(boxes) > 0:

                selected_idx = (
                    selected_idx + 1
                ) % len(boxes)

        # -----------------------------
        # SAVE
        # -----------------------------
        elif key == ord('y'):

            shutil.copy(
                img_path,
                os.path.join(output_images, img_name)
            )

            with open(label_path, "w") as f:

                for (x1, y1, x2, y2) in boxes:

                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h

                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    f.write(
                        f"0 {xc} {yc} {bw} {bh}\n"
                    )

            print(f"✅ Saved: {img_name}")

        # -----------------------------
        # NEXT IMAGE
        # -----------------------------
        elif key == ord('d'):

            i += 1
            break

        # -----------------------------
        # PREVIOUS IMAGE
        # -----------------------------
        elif key == ord('a'):

            i = max(0, i - 1)
            break

        # -----------------------------
        # RELOAD LABELS
        # -----------------------------
        elif key == ord('r'):

            boxes = load_labels(label_path, w, h)

            if boxes is None:
                boxes = []

            print("🔄 Reloaded")

        # -----------------------------
        # START ROI ZOOM
        # -----------------------------
        elif key == ord('z'):

            zoom_mode = True

            print("🔍 Drag mouse to select zoom area")

        # -----------------------------
        # RESET VIEW
        # -----------------------------
        elif key == ord('0'):

            zoom = 1.0

            view_x1 = 0
            view_y1 = 0
            view_x2 = 0
            view_y2 = 0

        # -----------------------------
        # QUIT
        # -----------------------------
        elif key == ord('q'):

            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

print("🎯 All done!")