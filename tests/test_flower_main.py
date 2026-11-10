import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOWER_MAIN_PATH = REPO_ROOT / "flower" / "main.py"


def load_flower_main_module():
    fake_flask = types.ModuleType("flask")

    class DummyFlask:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def route(self, *args, **kwargs):
            del args, kwargs

            def decorator(fn):
                return fn

            return decorator

        def run(self, *args, **kwargs):
            del args, kwargs

    fake_flask.Flask = DummyFlask
    fake_flask.jsonify = lambda payload: payload
    fake_flask.request = types.SimpleNamespace(args=types.SimpleNamespace(get=lambda key: None))

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.login = lambda *args, **kwargs: None

    original_modules = {
        "flask": sys.modules.get("flask"),
        "wandb": sys.modules.get("wandb"),
    }
    sys.modules["flask"] = fake_flask
    sys.modules["wandb"] = fake_wandb

    try:
        spec = importlib.util.spec_from_file_location("flower_main_under_test", FLOWER_MAIN_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class TestFlowerMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_flower_main_module()

    def test_startup_without_credentials_is_offline_and_never_logs_in(self):
        with patch.dict(self.module.os.environ, {}, clear=True), \
                patch.object(self.module.sys, "argv", ["main.py", "--insecure"]), \
                patch.object(self.module.wandb, "login") as login, \
                patch.object(self.module, "setup_flask"), \
                patch.object(self.module.subprocess, "run") as run:
            self.module.main()
            self.assertEqual(self.module.os.environ["WANDB_MODE"], "offline")
            login.assert_not_called()
            run.assert_called_once_with(["flower-superexec", "--plugin-type", "serverapp", "--insecure"])

    def test_explicit_local_modes_do_not_login_even_with_a_key(self):
        for mode in ("offline", "disabled"):
            with self.subTest(mode=mode), \
                    patch.dict(self.module.os.environ, {"WANDB_MODE": mode, "WANDB_API_KEY": "test-key"}, clear=True), \
                    patch.object(self.module.wandb, "login") as login, \
                    patch.object(self.module, "setup_flask"), \
                    patch.object(self.module.subprocess, "run") as run:
                self.module.main()
                login.assert_not_called()
                self.assertEqual(run.call_count, 1)

    def test_supplied_key_defaults_to_online_with_one_login(self):
        with patch.dict(self.module.os.environ, {"WANDB_API_KEY": "test-key"}, clear=True), \
                patch.object(self.module.wandb, "login") as login, \
                patch.object(self.module, "setup_flask"), \
                patch.object(self.module.subprocess, "run") as run:
            self.module.main()
            login.assert_called_once_with(key="test-key")
            self.assertEqual(self.module.os.environ["WANDB_MODE"], "online")
            self.assertEqual(run.call_count, 1)

    def test_invalid_or_uncredentialed_online_mode_fails_before_starting(self):
        for mode in ("online", "typo"):
            with self.subTest(mode=mode), \
                    patch.dict(self.module.os.environ, {"WANDB_MODE": mode}, clear=True), \
                    patch.object(self.module, "setup_flask") as flask, \
                    patch.object(self.module.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    self.module.main()
                flask.assert_not_called()
                run.assert_not_called()

    def test_normalize_payload_converts_run_ids_to_strings(self):
        payload = {
            "success": True,
            "run-id": 123,
            "runs": [{"run-id": 456}, {"run-id": "789"}],
        }

        normalized = self.module.normalize_payload(payload)

        self.assertEqual(normalized["run-id"], "123")
        self.assertEqual(normalized["runs"][0]["run-id"], "456")
        self.assertEqual(normalized["runs"][1]["run-id"], "789")

    def test_build_run_command_uses_repo_profile(self):
        self.assertEqual(
            self.module.build_run_command(),
            ["flwr", "run", ".", "deployment", "--format", "json"],
        )

    def test_build_ls_command_supports_optional_run_id(self):
        self.assertEqual(
            self.module.build_ls_command(),
            ["flwr", "ls", "deployment", "--format", "json"],
        )
        self.assertEqual(
            self.module.build_ls_command("run-123"),
            ["flwr", "ls", "--run-id", "run-123", "deployment", "--format", "json"],
        )

    def test_build_serverapp_command_normalizes_appio_arg_name(self):
        command = self.module.build_serverapp_command(
            ["--insecure", "--serverappio-api-address", "localhost:9091"]
        )

        self.assertEqual(
            command,
            [
                "flower-superexec",
                "--plugin-type",
                "serverapp",
                "--insecure",
                "--appio-api-address",
                "localhost:9091",
            ],
        )

    def test_run_flower_command_returns_success_payload(self):
        completed = types.SimpleNamespace(
            stdout='{"success": true, "run-id": 42, "runs": [{"run-id": 99}]}',
            stderr="",
            returncode=0,
        )

        with patch.object(self.module.subprocess, "run", return_value=completed) as run:
            payload, status = self.module.run_flower_command(["flwr", "ls"])

        self.assertEqual(status, 200)
        self.assertEqual(payload["run-id"], "42")
        self.assertEqual(payload["runs"][0]["run-id"], "99")
        run.assert_called_once()

    def test_run_flower_command_handles_empty_stdout(self):
        completed = types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

        with patch.object(self.module.subprocess, "run", return_value=completed):
            payload, status = self.module.run_flower_command(["flwr", "ls"])

        self.assertEqual(status, 503)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error-message"], "boom")
        self.assertEqual(payload["runs"], [])

    def test_run_flower_command_handles_invalid_json(self):
        completed = types.SimpleNamespace(stdout="not-json", stderr="bad json", returncode=0)

        with patch.object(self.module.subprocess, "run", return_value=completed):
            payload, status = self.module.run_flower_command(["flwr", "ls"])

        self.assertEqual(status, 503)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error-message"], "bad json")

    def test_run_flower_command_propagates_cli_failure_payload(self):
        completed = types.SimpleNamespace(
            stdout='{"success": false, "runs": [], "error-message": "unavailable"}',
            stderr="",
            returncode=0,
        )

        with patch.object(self.module.subprocess, "run", return_value=completed):
            payload, status = self.module.run_flower_command(["flwr", "ls"])

        self.assertEqual(status, 503)
        self.assertEqual(payload["error-message"], "unavailable")

    def test_run_flower_command_handles_oserror(self):
        with patch.object(self.module.subprocess, "run", side_effect=OSError("missing flwr")):
            payload, status = self.module.run_flower_command(["flwr", "ls"])

        self.assertEqual(status, 503)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error-message"], "missing flwr")


if __name__ == "__main__":
    unittest.main()
