import os
import sys
from io import StringIO

from keras_core.testing import test_case
from keras_core.utils.code_stats import count_loc


class TestCountLoc(test_case.TestCase):
    def setUp(self):
        self.test_dir = "test_directory"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def create_file(self, filename, content):
        with open(
            os.path.join(self.test_dir, filename), "w", encoding="utf-8"
        ) as f:
            f.write(content)

    def test_count_loc_valid_python(self):
        self.create_file(
            "sample.py", "# This is a test file\n\nprint('Hello')\n"
        )
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_exclude_test_files(self):
        self.create_file("sample_test.py", "print('Hello')\n")
        loc = count_loc(self.test_dir, exclude=("_test",))
        self.assertEqual(loc, 0)

    def test_other_extensions(self):
        self.create_file("sample.txt", "Hello\n")
        loc = count_loc(self.test_dir, extensions=(".py",))
        self.assertEqual(loc, 0)

    def test_comment_lines(self):
        self.create_file(
            "sample.py", "# Comment\nprint('Hello')\n# Another comment\n"
        )
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_multiline_strings(self):
        content = '"""\nMultiline str\nAnother line\n"""\nprint("Out str")\n'
        self.create_file("sample.py", content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_partial_multiline_strings(self):
        content = '"""\nMultiline string not ending in this file.'
        self.create_file("sample.py", content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)

    def test_empty_file(self):
        self.create_file("empty.py", "")
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)

    def test_whitespace_only(self):
        self.create_file("whitespace.py", "     \n\t\n")
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)

    def test_nested_multiline_strings(self):
        content = '''print("Start")
        """
        First level multiline
        """
        Second line outside any multiline
        """
        First level multiline
        """Some inline code here"""
        Nested multiline
        """
        print("End")
        '''
        self.create_file("nested_sample.py", content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 3)

    def test_inline_comments_after_code(self):
        content = 'print("Hello") # This is an inline comment'
        self.create_file("inline_comment_sample.py", content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)  # The comment shouldn't affect the count

    def test_directory_structure(self):
        content1 = 'print("Hello from file1")'
        content2 = 'print("Hello from file2")'
        os.mkdir(os.path.join(self.test_dir, "subdir"))
        self.create_file("sample1.py", content1)
        self.create_file(os.path.join("subdir", "sample2.py"), content2)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 2)  # Both files should be counted

    def test_normal_directory_name(self):
        content = 'print("Hello from a regular directory")'
        os.makedirs(os.path.join(self.test_dir, "some_test_dir"))
        self.create_file(os.path.join("some_test_dir", "sample.py"), content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)  # Should count normally

    def test_exclude_directory_name(self):
        content = 'print("Hello from an excluded directory")'
        os.makedirs(os.path.join(self.test_dir, "dir_test"))
        self.create_file(os.path.join("dir_test", "sample.py"), content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)
        # Shouldn't count the file in dir_test due to the exclusion pattern

    def test_verbose_output(self):
        content = 'print("Hello")'
        self.create_file("sample.py", content)
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        count_loc(self.test_dir, verbose=1)
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout
        self.assertIn("Count LoCs in", output)

    def test_multiline_string_same_line(self):
        content = '''"""This is a multiline string ending on the same line"""
        print("Outside string")'''
        self.create_file("same_line_multiline.py", content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)  # Only the print statement should count
