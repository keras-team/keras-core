import os


def count_loc(
    directory: str,
    exclude: tuple = ("_test",),
    extensions: tuple = (".py",),
    verbose: int = 0,
) -> int:
    """Count the Lines of Code (LOC) in a directory.

    This function traverses a specified directory and calculates the number of lines of code
    present in files with given extensions. It excludes lines from multiline strings,
    comments, and blank lines. The function also provides the ability to exclude certain
    files or directories.

    Args:
        directory (str): Path of the directory to count LOC.
        exclude (tuple): Patterns to exclude files or directories. Default is `("_test",)`.
        extensions (tuple): File extensions to consider for counting. Default is `(".py",)`.
        verbose (int): If set to 1, displays the files being processed. Defaults to 0.

    Returns:
        int: Total number of lines of code in the directory.
    """
    loc = 0
    for root, _, fnames in os.walk(directory):
        skip = False
        for ex in exclude:
            if root.endswith(ex):
                skip = True
        if skip:
            continue

        for fname in fnames:
            skip = False
            for ext in extensions:
                if not fname.endswith(ext):
                    skip = True
                    break
                for ex in exclude:
                    if fname.endswith(ex + ext):
                        skip = True
                        break
            if skip:
                continue

            fname = os.path.join(root, fname)
            if verbose:
                print(f"Count LoCs in {fname}")

            with open(fname) as f:
                lines = f.read().split("\n")

            string_open = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Start of a multiline string
                if line.startswith('"""') and not string_open:
                    string_open = True

                    # If the line ends the multiline string as well
                    if line.count('"""') == 2:
                        string_open = False
                        continue
                    else:
                        continue

                # End of a multiline string
                if line.endswith('"""') and string_open:
                    string_open = False
                    continue

                # Counting LOC when not inside a multiline string
                if not string_open:
                    loc += 1
    return loc
