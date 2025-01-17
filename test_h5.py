import h5py

def print_h5_keys_and_values(file_path):
    def recursively_print(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")
            print(f"Value: {obj[()]}\n")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    try:
        with h5py.File(file_path, 'r') as h5_file:
            print(f"Contents of HDF5 file: {file_path}\n")
            h5_file.visititems(recursively_print)
    except Exception as e:
        print(f"Error reading the file: {e}")

# Replace 'your_file.h5' with the path to your HDF5 file
file_path = "recordings/Demo_Recording/frames.h5"


if __name__ == "__main__":
    print_h5_keys_and_values(file_path)
    