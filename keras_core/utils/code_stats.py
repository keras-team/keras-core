import os


def count_loc(directory, exclude=("_test",), extensions=(".py",), verbose=0):
    """Count lines of code (LOC) within files in a specified directory.

    This function will recursively walk through a directory, inspecting files
    for actual lines of code while ignoring comments, whitespace, and lines
    inside multi-line strings.

    Args:
        directory (str): The root directory from which to start counting LOC.
        exclude (tuple): File or directory patterns to exclude from LOC count.
            Default is `("_test",)`.
        extensions (tuple): File extensions to include in LOC count.
            Default is `(".py",)`.
        verbose (int): If set to 1, the function will print the names of
            the files being analyzed. Default is 0.

    Returns:
        int: The total lines of actual code in the directory.
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
