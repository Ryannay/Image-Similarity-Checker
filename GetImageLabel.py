import os
import pickle

def generate_label_dict(img_root="cifar_images", save_as="labels.pkl"):
    label_dict = {}
    class_to_idx = {}
    
    # Sorted to keep class indices consistent
    class_names = sorted(os.listdir(img_root))
    
    # Create a mapping: class name -> index
    for idx, class_name in enumerate(class_names):
        class_to_idx[class_name] = idx
        class_folder = os.path.join(img_root, class_name)
        
        for filename in os.listdir(class_folder):
            if filename.endswith(".png"):
                full_path = os.path.join(class_name, filename)  # Save relative path
                label_dict[full_path] = idx

    # Save label dict
    with open(save_as, "wb") as f:
        pickle.dump(label_dict, f)

    # Also return class mapping (optional)
    return label_dict, class_to_idx


label_dict, class_to_idx = generate_label_dict("D:\personal\stress\Practice_Project\Pytorch_Template\Practice_Dataset\cifar_images")
if __name__ == "__main__":
    print("Sample entry:", list(label_dict.items())[:5])
    print("Class mapping:", class_to_idx)

