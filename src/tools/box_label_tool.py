import cv2
import json
import os

# Global variables to store coordinates and state
top_left = (-1, -1)
bottom_right = (-1, -1)
drawing = False
box_preview = False
boxes = []


# Mouse callback function
def draw_box(event, x, y, flags, param):
    global top_left, bottom_right, drawing, box_preview, image, clone, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            # First click: set the start point and mark as drawing
            top_left = (x, y)
            drawing = True
            box_preview = False
        else:
            # Second click: set the end point and stop drawing
            bottom_right = (x, y)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 1)
            cv2.imshow("image", image)
            drawing = False
            box_preview = False
            boxes.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the preview of the box
            clone = image.copy()
            box_preview = True
            cv2.rectangle(clone, top_left, (x, y), (0, 255, 0), 1)
            cv2.imshow("image", clone)


# Directory with images
image_dir = "../unlabelled_boxes/"  # Adjust path to your images

# Directory to save box labels
box_labels_dir = "../box_labels/"
os.makedirs(box_labels_dir, exist_ok=True)

# List all files in the directory
image_files = os.listdir(image_dir)

# Process each file
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    clone = image.copy()

    # Create a window and bind the mouse callback function to it
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_box)

    # Main loop
    while True:
        if not box_preview:  # Only display the unmodified image if not previewing a box
            cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to exit the loop
        if key == ord("q"):
            break

    # Save the box coordinates
    print("Box coordinates:")
    file_name, _ = os.path.splitext(image_file)
    with open(os.path.join(box_labels_dir, f"{file_name}_boxes.json"), "w") as f:
        json.dump(boxes, f)

    # Clean up
    cv2.destroyAllWindows()
