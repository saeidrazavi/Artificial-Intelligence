import os


def get_files(main_dir='asl_dataset/asl_dataset'):
    files_addr, lbls = [], []
    for dir in os.listdir(main_dir):
        lbl = int(dir) if dir.isdigit() else (ord(dir) - 87)
        addrs = [
            f'{main_dir}/{dir}/{file_name}' for file_name in os.listdir(f'{main_dir}/{dir}')]
        files_addr.extend(addrs)
        lbls.extend([lbl]*len(addrs))
    return files_addr, lbls
