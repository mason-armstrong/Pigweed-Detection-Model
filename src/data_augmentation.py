#Data augumentation for pigweed dataset
import cv2
import os
import math

def read_annotations(txt_path):
    """
    Reads the annotations from a .txt file and returns a list of bounding boxes.

    Args:
        txt_path (str): The path to the .txt file containing the annotations.

    Returns:
        list: A list of bounding boxes, where each bounding box is represented as a list of [class_id, center_x, center_y, width, height].
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        bounding_boxes = []
        for line in lines:
            values = line.split()
            class_id, center_x, center_y, width, height = map(float, values)
            bounding_boxes.append([class_id, center_x, center_y, width, height])
        return bounding_boxes
    
def rotate(image, angle):
    """
    Rotates an image by a given angle.

    Args:
        image (numpy.ndarray): The image to be rotated.
        angle (float): The angle to rotate the image by.

    Returns:
        numpy.ndarray: The rotated image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

#Function to rotate bounding boxes: 
def rotate_bounding_boxes(bounding_boxes, angle, image_width, image_height):
            """
            Rotates a list of bounding boxes by a given angle.

            Args:
                bounding_boxes (list): A list of bounding boxes, where each bounding box is represented as a list of [class_id, center_x, center_y, width, height].
                angle (float): The angle to rotate the bounding boxes by.
                image_width (int): The width of the image.
                image_height (int): The height of the image.

            Returns:
                list: A list of rotated bounding boxes, where each bounding box is represented as a list of [class_id, center_x, center_y, width, height].
            """
            rotated_bounding_boxes = []
            for box in bounding_boxes:
                class_id, center_x, center_y, width, height = box
                # Convert the center coordinates to the origin
                center_x -= image_width / 2
                center_y -= image_height / 2
                # Rotate the center coordinates
                new_center_x = center_x * math.cos(math.radians(angle)) - center_y * math.sin(math.radians(angle))
                new_center_y = center_x * math.sin(math.radians(angle)) + center_y * math.cos(math.radians(angle))
                # Convert the center coordinates back to the original coordinate system
                new_center_x += image_width / 2
                new_center_y += image_height / 2
                # Add the rotated bounding box to the list
                rotated_bounding_boxes.append([class_id, new_center_x, new_center_y, width, height])
            return rotated_bounding_boxes
        
def save_annotations(bounding_boxes, txt_path):
            """
            Saves a list of bounding boxes to a text file.

            Args:
                bounding_boxes (list): A list of bounding boxes, where each bounding box is represented as a list of [class_id, center_x, center_y, width, height].
                txt_path (str): The path to the text file to save the bounding boxes to.
            """
            with open(txt_path, 'w') as f:
                for box in bounding_boxes:
                    class_id, center_x, center_y, width, height = box
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
                    
#Function to rotate a single image:
def rotate_image(image_path, save_path, angle):
    image = cv2.imread(image_path)
    if image is None:
        print("Could not read image")
        exit()
    h, w, _ = image.shape
    #Rotate the image
    rotated_image = rotate(image, angle)
    
    #Save image
    cv2.imwrite(save_path, rotated_image)
    
    
    
#Function to rotate images:
def rotate_images(image_folder, annotation_folder, output_folder, angle):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i in range(1, os.listdir(image_folder).__len__() + 1):
        image_name = f"pigweed_{i:03d}.png"
        txt_name = f"pigweed_{i:03d}.txt"
        
        image_path = os.path.join(image_folder, image_name)
        annotation_path = os.path.join(annotation_folder, txt_name)
        save_path = os.path.join(output_folder, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            print("Could not read image")
            exit()
        h, w, _ = image.shape

       
def draw_annotations(img, bounding_boxes, save_path):
    """
    Draws bounding boxes on an image and saves the annotated image to a file.

    Args:
        img (str): The path to the image file.
        bounding_boxes (list): A list of bounding boxes, where each bounding box is represented as a list of [class_id, center_x, center_y, width, height].
        save_path (str): The path to save the annotated image.

    Returns:
        None
    """
    image = cv2.imread(img)
    if image is None:
        print("Could not read image")
        exit()
    h, w, _ = image.shape
    for bounding_box in bounding_boxes:
        class_id, center_x, center_y, width, height = bounding_box
        x1 = int((center_x - width / 2) * w)
        y1 = int((center_y - height / 2) * h)
        x2 = int((center_x + width / 2) * w)
        y2 = int((center_y + height / 2) * h)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
    cv2.imwrite(save_path, image)
    
# Function to get the next index for saving images
def get_next_index(folder):
    existing_files = os.listdir(folder)
    indices = [int(file.split('_')[1].split('.')[0]) for file in existing_files if 'pigweed' in file]
    return max(indices, default=0) + 1


#Function to rename image based on number of images already in directory:
def rename_images(image_folder):
    image_names = os.listdir(image_folder)
    for i, image_name in enumerate(image_names):
        os.rename(os.path.join(image_folder, image_name), os.path.join(image_folder,f"pigweed_{i:03d}.png"))
    print("Done renaming images")

    

#Loop through directory containing annotated images:
image_folder = 'data\\raw\\images\\pigweed'
annotation_folder = 'Training Data\PigweedDataSet\\annotations'
output_folder = "Training Data\PigweedDataSet\\rotated_images"



# Iterate through directory containing original images, and rotate each image, and save it to the rotated_images folder:
next_index = get_next_index(output_folder)

for i in range(1, len(os.listdir(image_folder)) + 1):
        image_name = f"pigweed_{i:03d}.png"
        image_path = os.path.join(image_folder, image_name)

        for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            save_path = os.path.join(output_folder, f"pigweed_{next_index:03d}.png")
            rotate_image(image_path, save_path, angle)
            next_index += 1

    
    
    
    








