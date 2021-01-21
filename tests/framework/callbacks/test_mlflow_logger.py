import os
from unittest import TestCase
from unittest.mock import patch

from poutyne.framework.callbacks.mlflow_logger import _get_git_commit


class GetGitCommit(TestCase):
    def setUp(self) -> None:
        self.a_fake_path = "a_fake_path"

    @patch("poutyne.framework.mlflow_logger.git", None)
    def test_whenGitNotInstall_givenARepositoryPathToGetGitCommit_thenRaiseWarning(self):
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_fake_path)

    @patch("poutyne.framework.mlflow_logger.git", None)
    def test_whenGitNotInstall_givenARepositoryPathToGetGitCommit_thenGitCommitIsNone(self):
        self.assertIsNone(_get_git_commit(self.a_fake_path))

    def test_whenGitInstalled_givenAWrongRepositoryPathToGetGitCommit_thenRaiseWarning(self):
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_fake_path)

    # def test_whenGitInstalled_givenARepositoryPathToGetGitCommitButNotAGitRepo_thenRaiseWarning(self):
    #     os.makedirs(self.a_fake_path)
    #     with self.assertWarns(UserWarning):
    #         _get_git_commit(self.a_fake_path)
    #
    #     os.rmdir(self.a_fake_path)