import os
from matplotlib import pyplot as plt
from pathlib import Path


def build_path(*folders, file_name=None):
    """
    Returns the current directory appended with any additional folder names provided,
    and optionally appends a file name at the end.

    Parameters:
    *folders (str): A variable number of folder names to append to the current directory.
    file_name (str, optional): The file name to append to the path.

    Returns:
    str: The resulting full path.
    """
    # Start with the current directory and add any additional folders
    path = os.path.join(os.getcwd(), *folders)
    # If a file name is provided, append it to the path
    if file_name:
        path = os.path.join(path, file_name)
    return path


def comparison_histograms(data1, data2, bins, label1, label2, title, show=False, export=False, path=None):
    plt.clf()
    plt.hist(data1, bins=bins, alpha=0.5, label=label1)
    plt.hist(data2, bins=bins, alpha=0.5, label=label2)

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if show:
        plt.show()
    elif export:
        plt.savefig(build_path(path))


def ensure_directory(path_str):
    path = Path(path_str)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")
    return str(path)
