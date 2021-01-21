from unittest import TestCase
from unittest.mock import patch

import git

from poutyne.framework.callbacks.mlflow_logger import _get_git_commit


class GetGitCommit(TestCase):

    def setUp(self) -> None:
        self.a_fake_path = "a_fake_path"
        self.a_wrong_path = "/a_wrong_path"
        self.a_git_sha = "9bff900c30e80c3a35388d3e617db5b7a64c9afd"

    @patch("poutyne.framework.mlflow_logger.git", None)
    def test_whenGitNotInstall_givenARepositoryPathToGetGitCommit_thenRaiseWarning(self):
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_fake_path)

    @patch("poutyne.framework.mlflow_logger.git", None)
    def test_whenGitNotInstall_givenARepositoryPathToGetGitCommit_thenGitCommitIsNone(self):
        self.assertIsNone(_get_git_commit(self.a_fake_path))

    @patch("poutyne.framework.mlflow_logger.git.Repo")
    def test_whenGitInstalled_givenARepositoryPathToGetGitCommitButNotAGitRepo_thenRaiseWarning(self, git_repo_patch):
        git_repo_patch.side_effect = git.NoSuchPathError()
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_fake_path)

    def test_whenGitInstalled_givenAWrongRepositoryPathToGetGitCommit_thenRaiseWarning(self):
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_wrong_path)

    @patch("poutyne.framework.mlflow_logger.git.Repo")
    def test_whenGitInstalled_givenARepositoryPathToGetGitCommitAndAGitRepo_thenReturnCommit(self, git_repo_patch):
        git_repo_patch.return_value.head.commit.hexsha = self.a_git_sha

        actual = _get_git_commit(self.a_fake_path)
        expected = self.a_git_sha

        self.assertEqual(expected, actual)
