import os

#Function to rename all images in a folder
def rename_images(image_folder):
    image_names = os.listdir(image_folder)
    for i, image_name in enumerate(image_names):
        os.rename(os.path.join(image_folder, image_name), os.path.join(image_folder,f"not_pigweed_{i:03d}.png"))
    print("Done renaming images")
    
    
if __name__ == '__main__':
    rename_images('data/raw/images/not_pigweed')