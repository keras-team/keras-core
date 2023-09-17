import os
import tarfile
import urllib
import zipfile

from keras_core.testing import test_case
from keras_core.utils import file_utils


class TestGetFile(test_case.TestCase):
    def test_get_file_and_validate_it(self):
        """Tests get_file from a url, plus extraction and validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()

        text_file_path = os.path.join(orig_dir, "test.txt")
        zip_file_path = os.path.join(orig_dir, "test.zip")
        tar_file_path = os.path.join(orig_dir, "test.tar.gz")

        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with tarfile.open(tar_file_path, "w:gz") as tar_file:
            tar_file.add(text_file_path)

        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            zip_file.write(text_file_path)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(tar_file_path)),
        )

        path = file_utils.get_file(
            "test.txt", origin, untar=True, cache_subdir=dest_dir
        )
        filepath = path + ".tar.gz"
        hashval_sha256 = file_utils.hash_file(filepath)
        hashval_md5 = file_utils.hash_file(filepath, algorithm="md5")
        path = file_utils.get_file(
            "test.txt",
            origin,
            md5_hash=hashval_md5,
            untar=True,
            cache_subdir=dest_dir,
        )
        path = file_utils.get_file(
            filepath,
            origin,
            file_hash=hashval_sha256,
            extract=True,
            cache_subdir=dest_dir,
        )
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(file_utils.validate_file(filepath, hashval_sha256))
        self.assertTrue(file_utils.validate_file(filepath, hashval_md5))
        os.remove(filepath)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(zip_file_path)),
        )

        hashval_sha256 = file_utils.hash_file(zip_file_path)
        hashval_md5 = file_utils.hash_file(zip_file_path, algorithm="md5")
        path = file_utils.get_file(
            "test",
            origin,
            md5_hash=hashval_md5,
            extract=True,
            cache_subdir=dest_dir,
        )
        path = file_utils.get_file(
            "test",
            origin,
            file_hash=hashval_sha256,
            extract=True,
            cache_subdir=dest_dir,
        )
        self.assertTrue(os.path.exists(path))
        self.assertTrue(file_utils.validate_file(path, hashval_sha256))
        self.assertTrue(file_utils.validate_file(path, hashval_md5))
        os.remove(path)

        for file_path, extract in [
            (text_file_path, False),
            (tar_file_path, True),
            (zip_file_path, True),
        ]:
            origin = urllib.parse.urljoin(
                "file://",
                urllib.request.pathname2url(os.path.abspath(file_path)),
            )
            hashval_sha256 = file_utils.hash_file(file_path)
            path = file_utils.get_file(
                origin=origin,
                file_hash=hashval_sha256,
                extract=extract,
                cache_subdir=dest_dir,
            )
            self.assertTrue(os.path.exists(path))
            self.assertTrue(file_utils.validate_file(path, hashval_sha256))
            os.remove(path)

        with self.assertRaisesRegexp(
            ValueError, 'Please specify the "origin".*'
        ):
            _ = file_utils.get_file()

    def test_get_file_with_tgz_extension(self):
        """Tests get_file from a url, plus extraction and validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = dest_dir

        text_file_path = os.path.join(orig_dir, "test.txt")
        tar_file_path = os.path.join(orig_dir, "test.tar.gz")

        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with tarfile.open(tar_file_path, "w:gz") as tar_file:
            tar_file.add(text_file_path)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(tar_file_path)),
        )

        path = file_utils.get_file(
            "test.txt.tar.gz", origin, untar=True, cache_subdir=dest_dir
        )
        self.assertTrue(path.endswith(".txt"))
        self.assertTrue(os.path.exists(path))

    def test_get_file_with_integrity_check(self):
        """Tests get_file with validation before download."""
        orig_dir = self.get_temp_dir()
        file_path = os.path.join(orig_dir, "test.txt")

        with open(file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        hashval = file_utils.hash_file(file_path)

        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(file_path))
        )

        path = file_utils.get_file("test.txt", origin, file_hash=hashval)
        self.assertTrue(os.path.exists(path))

    def test_get_file_with_failed_integrity_check(self):
        """Tests get_file with validation before download."""
        orig_dir = self.get_temp_dir()
        file_path = os.path.join(orig_dir, "test.txt")

        with open(file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        hashval = "0" * 64

        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(file_path))
        )

        with self.assertRaisesRegex(
            ValueError, "Incomplete or corrupted file.*"
        ):
            _ = file_utils.get_file("test.txt", origin, file_hash=hashval)

    def test_is_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("gs://bucket/path"))
        self.assertTrue(file_utils.is_remote_path("http://example.com/path"))
        self.assertFalse(file_utils.is_remote_path("/local/path"))
        self.assertFalse(file_utils.is_remote_path("./relative/path"))

    def test_exists(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_exists.txt")

        with open(file_path, "w") as f:
            f.write("test")

        self.assertTrue(file_utils.exists(file_path))
        self.assertFalse(
            file_utils.exists(os.path.join(temp_dir, "non_existent.txt"))
        )

    def test_file_open_read(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_file.txt")
        content = "test content"

        with open(file_path, "w") as f:
            f.write(content)

        with file_utils.File(file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_file_open_write(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_file_write.txt")
        content = "test write content"

        with file_utils.File(file_path, "w") as f:
            f.write(content)

        with open(file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_isdir(self):
        temp_dir = self.get_temp_dir()
        self.assertTrue(file_utils.isdir(temp_dir))

        file_path = os.path.join(temp_dir, "test_isdir.txt")
        with open(file_path, "w") as f:
            f.write("test")
        self.assertFalse(file_utils.isdir(file_path))

    def test_join_simple(self):
        self.assertEqual(file_utils.join("/path", "to", "dir"), "/path/to/dir")

    def test_join_single_directory(self):
        self.assertEqual(file_utils.join("/path"), "/path")

    def setUp(self):
        self.temp_dir = self.get_temp_dir()
        self.file_path = os.path.join(self.temp_dir, "sample_file.txt")
        with open(self.file_path, "w") as f:
            f.write("Sample content")

    def test_is_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("gcs://bucket/path"))
        self.assertFalse(file_utils.is_remote_path("/local/path"))

    def test_exists(self):
        self.assertTrue(file_utils.exists(self.file_path))
        self.assertFalse(file_utils.exists("/path/that/does/not/exist"))

    def test_isdir(self):
        self.assertTrue(file_utils.isdir(self.temp_dir))
        self.assertFalse(file_utils.isdir(self.file_path))

    def test_listdir(self):
        content = file_utils.listdir(self.temp_dir)
        self.assertIn("sample_file.txt", content)

    def test_makedirs_and_rmtree(self):
        new_dir = os.path.join(self.temp_dir, "new_directory")
        file_utils.makedirs(new_dir)
        self.assertTrue(os.path.isdir(new_dir))
        file_utils.rmtree(new_dir)
        self.assertFalse(os.path.exists(new_dir))

    def test_copy(self):
        dest_path = os.path.join(self.temp_dir, "copy_sample_file.txt")
        file_utils.copy(self.file_path, dest_path)
        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "Sample content")

    def test_file_open_read(self):
        with file_utils.File(self.file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "Sample content")

    def test_file_open_write(self):
        with file_utils.File(self.file_path, "w") as f:
            f.write("New content")
        with open(self.file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "New content")

    def test_remove_sub_directory(self):
        parent_dir = os.path.join(self.get_temp_dir(), "parent_directory")
        child_dir = os.path.join(parent_dir, "child_directory")
        file_utils.makedirs(child_dir)
        file_utils.rmtree(parent_dir)
        self.assertFalse(os.path.exists(parent_dir))
        self.assertFalse(os.path.exists(child_dir))

    def test_remove_files_inside_directory(self):
        dir_path = os.path.join(self.get_temp_dir(), "test_directory")
        file_path = os.path.join(dir_path, "test.txt")
        file_utils.makedirs(dir_path)
        with open(file_path, "w") as f:
            f.write("Test content")
        file_utils.rmtree(dir_path)
        self.assertFalse(os.path.exists(dir_path))
        self.assertFalse(os.path.exists(file_path))

    def test_handle_complex_paths(self):
        complex_dir = os.path.join(self.get_temp_dir(), "complex dir@#%&!")
        file_utils.makedirs(complex_dir)
        file_utils.rmtree(complex_dir)
        self.assertFalse(os.path.exists(complex_dir))
