import os
def add_prefix(prefix, curr_dir):
    # dirs = os.listdir(curr_dir)
    files = [entry.name for entry in os.scandir(curr_dir) if entry.is_file()]
    dirs = [entry.name for entry in os.scandir(curr_dir) if entry.is_dir()]
    for d in dirs:
        folder_name = d.split("/")[-1]
        new_d = curr_dir + "/" + d
        add_prefix(folder_name, new_d)
    for file in files:
        new_prefix = curr_dir.split("/")[-1] if prefix is None else prefix
        new_file = new_prefix + file
        if not file.startswith(new_prefix):
            f = curr_dir + "/" + file
            new_f = curr_dir + "/" + new_file
            print("file: ", file)
            print("new_file: ", new_file)
            print("f: ", f)
            print("new_f: ", new_f)
            os.rename(f, new_f)
if __name__ == '__main__':
    prefix = None
    curr_dir = "/Volumes/passp_white1/我的移动硬盘/视频音频/短视频/123"
    add_prefix(prefix, curr_dir)

