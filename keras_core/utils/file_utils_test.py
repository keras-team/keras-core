import os
import pathlib
import shutil
import tarfile
import tempfile
import urllib
import zipfile

from keras_core.testing import test_case
from keras_core.utils import file_utils


class PathToStringTest(test_case.TestCase):
    def test_path_to_string_with_string_path(self):
        path = "/path/to/file.txt"
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, path)

    def test_path_to_string_with_PathLike_object(self):
        path = pathlib.Path("/path/to/file.txt")
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, str(path))

    def test_path_to_string_with_non_string_typed_path_object(self):
        class NonStringTypedPathObject:
            def __fspath__(self):
                return "/path/to/file.txt"

        path = NonStringTypedPathObject()
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, "/path/to/file.txt")

    def test_path_to_string_with_none_path(self):
        string_path = file_utils.path_to_string(None)
        self.assertEqual(string_path, None)


class ResolvePathTest(test_case.TestCase):
    def test_resolve_path_with_absolute_path(self):
        path = "/path/to/file.txt"
        resolved_path = file_utils.resolve_path(path)
        self.assertEqual(resolved_path, os.path.realpath(os.path.abspath(path)))

    def test_resolve_path_with_relative_path(self):
        path = "./file.txt"
        resolved_path = file_utils.resolve_path(path)
        self.assertEqual(resolved_path, os.path.realpath(os.path.abspath(path)))


class IsPathInDirTest(test_case.TestCase):
    def test_is_path_in_dir_with_absolute_paths(self):
        base_dir = "/path/to/base_dir"
        path = "/path/to/base_dir/file.txt"
        self.assertTrue(file_utils.is_path_in_dir(path, base_dir))


class IsLinkInDirTest(test_case.TestCase):
    def setUp(self):
        # This setup method runs before each test.
        # Ensuring both base directories are clean before the tests are run.
        self._cleanup("test_path/to/base_dir")
        self._cleanup("./base_dir")

    def _cleanup(self, base_dir):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_is_link_in_dir_with_absolute_paths(self):
        base_dir = "test_path/to/base_dir"
        link_path = os.path.join(base_dir, "symlink")
        target_path = os.path.join(base_dir, "file.txt")

        # Create the base_dir directory if it does not exist.
        os.makedirs(base_dir, exist_ok=True)

        # Create the file.txt file.
        with open(target_path, "w") as f:
            f.write("Hello, world!")

        os.symlink(target_path, link_path)

        # Creating a stat_result-like object with a name attribute
        info = os.lstat(link_path)
        info = type(
            "stat_with_name",
            (object,),
            {
                "name": os.path.basename(link_path),
                "linkname": os.readlink(link_path),
            },
        )

        self.assertTrue(file_utils.is_link_in_dir(info, base_dir))

    def test_is_link_in_dir_with_relative_paths(self):
        base_dir = "./base_dir"
        link_path = os.path.join(base_dir, "symlink")
        target_path = os.path.join(base_dir, "file.txt")

        # Create the base_dir directory if it does not exist.
        os.makedirs(base_dir, exist_ok=True)

        # Create the file.txt file.
        with open(target_path, "w") as f:
            f.write("Hello, world!")

        os.symlink(target_path, link_path)

        # Creating a stat_result-like object with a name attribute
        info = os.lstat(link_path)
        info = type(
            "stat_with_name",
            (object,),
            {
                "name": os.path.basename(link_path),
                "linkname": os.readlink(link_path),
            },
        )

        self.assertTrue(file_utils.is_link_in_dir(info, base_dir))

    def tearDown(self):
        self._cleanup("test_path/to/base_dir")
        self._cleanup("./base_dir")


class TestFilterSafePaths(test_case.TestCase):
    def setUp(self):
        # Assuming the temp directory is the base dir for our tests
        self.base_dir = os.path.join(os.getcwd(), "temp_dir")
        os.makedirs(self.base_dir, exist_ok=True)
        self.tar_path = os.path.join(self.base_dir, "test.tar")

    def tearDown(self):
        os.remove(self.tar_path)
        os.rmdir(self.base_dir)

    def test_member_within_base_dir(self):
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(
                __file__, arcname="safe_path.txt"
            )  # Adds this test file to the tar archive
        with tarfile.open(self.tar_path, "r") as tar:
            members = list(file_utils.filter_safe_paths(tar.getmembers()))
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "safe_path.txt")


class ExtractArchiveTest(test_case.TestCase):
    def setUp(self):
        """Create temporary directories and files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_content = "Hello, world!"

        # Create sample files to be archived
        with open(os.path.join(self.temp_dir, "sample.txt"), "w") as f:
            f.write(self.file_content)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    def create_tar(self):
        archive_path = os.path.join(self.temp_dir, "sample.tar")
        with tarfile.open(archive_path, "w") as archive:
            archive.add(
                os.path.join(self.temp_dir, "sample.txt"), arcname="sample.txt"
            )
        return archive_path

    def create_zip(self):
        archive_path = os.path.join(self.temp_dir, "sample.zip")
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(
                os.path.join(self.temp_dir, "sample.txt"), arcname="sample.txt"
            )
        return archive_path

    def test_extract_tar(self):
        archive_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "extract_tar")
        result = file_utils.extract_archive(archive_path, extract_path, "tar")
        self.assertTrue(result)
        with open(os.path.join(extract_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_extract_zip(self):
        archive_path = self.create_zip()
        extract_path = os.path.join(self.temp_dir, "extract_zip")
        result = file_utils.extract_archive(archive_path, extract_path, "zip")
        self.assertTrue(result)
        with open(os.path.join(extract_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_extract_auto(self):
        # This will test the 'auto' functionality
        tar_archive_path = self.create_tar()
        zip_archive_path = self.create_zip()

        extract_tar_path = os.path.join(self.temp_dir, "extract_auto_tar")
        extract_zip_path = os.path.join(self.temp_dir, "extract_auto_zip")

        self.assertTrue(
            file_utils.extract_archive(tar_archive_path, extract_tar_path)
        )
        self.assertTrue(
            file_utils.extract_archive(zip_archive_path, extract_zip_path)
        )

        with open(os.path.join(extract_tar_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

        with open(os.path.join(extract_zip_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)


class TestHashFile(test_case.TestCase):
    def setUp(self):
        self.test_content = b"Hello, World!"
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(self.test_content)
        self.temp_file.close()

    def tearDown(self):
        os.remove(self.temp_file.name)

    def test_hash_file_sha256(self):
        expected_sha256 = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )
        calculated_sha256 = file_utils.hash_file(
            self.temp_file.name, algorithm="sha256"
        )
        self.assertEqual(expected_sha256, calculated_sha256)

    def test_hash_file_md5(self):
        expected_md5 = "65a8e27d8879283831b664bd8b7f0ad4"
        calculated_md5 = file_utils.hash_file(
            self.temp_file.name, algorithm="md5"
        )
        self.assertEqual(expected_md5, calculated_md5)


class TestValidateFile(test_case.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False)
        self.tmp_file.write(b"Hello, World!")
        self.tmp_file.close()

        self.sha256_hash = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )
        self.md5_hash = "65a8e27d8879283831b664bd8b7f0ad4"

    def test_validate_file_sha256(self):
        self.assertTrue(
            file_utils.validate_file(
                self.tmp_file.name, self.sha256_hash, "sha256"
            )
        )

    def test_validate_file_md5(self):
        self.assertTrue(
            file_utils.validate_file(self.tmp_file.name, self.md5_hash, "md5")
        )

    def test_validate_file_auto_sha256(self):
        self.assertTrue(
            file_utils.validate_file(
                self.tmp_file.name, self.sha256_hash, "auto"
            )
        )

    def test_validate_file_auto_md5(self):
        self.assertTrue(
            file_utils.validate_file(self.tmp_file.name, self.md5_hash, "auto")
        )

    def test_validate_file_wrong_hash(self):
        wrong_hash = "deadbeef" * 8  #
        self.assertFalse(
            file_utils.validate_file(self.tmp_file.name, wrong_hash, "sha256")
        )

    def tearDown(self):
        os.remove(self.tmp_file.name)


class TestIsRemotePath(test_case.TestCase):
    def test_gcs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/gcs/some/path/to/file.txt"))
        self.assertTrue(file_utils.is_remote_path("/gcs/another/directory/"))

    def test_cns_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/cns/some/path/to/file.txt"))
        self.assertTrue(file_utils.is_remote_path("/cns/another/directory/"))

    def test_cfs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/cfs/some/path/to/file.txt"))
        self.assertTrue(file_utils.is_remote_path("/cfs/another/directory/"))

    def test_http_remote_path(self):
        self.assertTrue(
            file_utils.is_remote_path("http://example.com/path/to/file.txt")
        )
        self.assertTrue(
            file_utils.is_remote_path("https://secure.example.com/directory/")
        )
        self.assertTrue(
            file_utils.is_remote_path("ftp://files.example.com/somefile.txt")
        )

    def test_non_remote_paths(self):
        self.assertFalse(file_utils.is_remote_path("/local/path/to/file.txt"))
        self.assertFalse(
            file_utils.is_remote_path("C:\\local\\path\\on\\windows\\file.txt")
        )
        self.assertFalse(file_utils.is_remote_path("~/relative/path/"))
        self.assertFalse(file_utils.is_remote_path("./another/relative/path"))

    def test_edge_cases(self):
        self.assertFalse(file_utils.is_remote_path(""))
        self.assertFalse(file_utils.is_remote_path(None))
        self.assertFalse(file_utils.is_remote_path(12345))


class TestGetFile(test_case.TestCase):
    def setUp(self):
        self.temp_dir = self.get_temp_dir()
        self.file_path = os.path.join(self.temp_dir, "sample_file.txt")
        with open(self.file_path, "w") as f:
            f.write("Sample content")

    def test_get_file_and_validate_it(self):
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        text_file_path = os.path.join(orig_dir, "test.txt")
        tar_file_path = os.path.join(orig_dir, "test.tar.gz")
        zip_file_path = os.path.join(orig_dir, "test.zip")

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
