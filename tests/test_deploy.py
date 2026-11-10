"""Exercise deployment rendering and reruns without contacting a cluster."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
KUBECTL = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\n')
if args[:2] == ['config', 'current-context']:
    print('test')
elif args[:2] == ['config', 'view']:
    print('https://test.invalid')
elif args[:2] == ['get', 'nodes']:
    print(json.dumps({'items': [{'metadata': {'labels': {'fl-role': role}}, 'status': {'addresses': [{'type': 'InternalIP', 'address': '192.0.2.1'}]}} for role in ['server', 'client']]}))
elif args[0] == 'get':
    if args[1:3] == ['secret', 'server-cert'] and os.environ.get('MISSING_CERT'):
        sys.exit(1)
elif args[0] == 'create':
    if '-f' in args:
        name = args[args.index('-f') + 1].split('/')[-1]
        names = {'server-deployment.yaml': ['flower-serverapp', 'server-logic', 'flower-superlink'], 'orchestrator-daemonset.yaml': ['orchestrator'], 'trainer-daemonset.yaml': ['trainer-logic', 'flower-clientapp']}.get(name, ['other'])
        print(json.dumps({'spec': {'template': {'spec': {'containers': [{'name': n, 'image': 'original'} for n in names]}}}}))
    else:
        print('{}')
elif args[0] == 'apply' and args[-1] == '-':
    value = sys.stdin.read()
    with open(os.environ['APPLY_LOG'], 'a') as log:
        log.write(json.dumps(json.loads(value)) + '\n')
'''


class TestDeploy(unittest.TestCase):
    def run_deploy(self, directory, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('WANDB_', 'IMAGE_', 'API_PORT', 'ROLLOUT_TIMEOUT'))}
        env.update(PATH=str(directory) + os.pathsep + env['PATH'], CALL_LOG=str(directory / 'calls'), APPLY_LOG=str(directory / 'applied'))
        env.update(overrides)
        return subprocess.run(['bash', str(ROOT / 'deploy.sh'), '--yes', '--kubeconfig', str(directory / 'config'), '--image-tag', 'sha-test', '--api-port', '8081'], env=env, text=True, capture_output=True)

    def prepare(self, directory):
        (directory / 'config').touch()
        (directory / 'kubectl').write_text(KUBECTL)
        (directory / 'kubectl').chmod(0o755)

    def test_repeat_deployment_preserves_secrets_and_renders_before_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.prepare(directory)
            for _ in range(2):
                result = self.run_deploy(directory)
                self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in (directory / 'calls').read_text().splitlines()]
            self.assertFalse(any(c[0] == 'delete' for c in calls))
            self.assertFalse(any(c[:2] == ['create', 'secret'] for c in calls))
            rollouts = [c for c in calls if c[:2] == ['rollout', 'status']]
            self.assertEqual(len(rollouts), 14)
            self.assertTrue(all('--timeout=300s' in c for c in rollouts))
            documents = [json.loads(line) for line in (directory / 'applied').read_text().splitlines()]
            containers = [c for d in documents if 'spec' in d for c in d['spec']['template']['spec']['containers']]
            for name, variable, value in [('flower-serverapp', 'WANDB_MODE', 'offline'), ('server-logic', 'LISTEN_PORT', '8081'), ('orchestrator', 'SERVER_PORT', '8081')]:
                matches = [c for c in containers if c['name'] == name]
                self.assertEqual(len(matches), 2)
                for container in matches:
                    self.assertEqual(container['image'], f'ghcr.io/idlab-discover/aide-fl/{name}:sha-test')
                    self.assertIn({'name': variable, 'value': value}, container['env'])

    def test_incomplete_certificates_fail_without_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.prepare(directory)
            result = self.run_deploy(directory, MISSING_CERT='1')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Incomplete Flower certificates', result.stderr)
            calls = [json.loads(line) for line in (directory / 'calls').read_text().splitlines()]
            self.assertFalse(any(c[:2] == ['create', 'secret'] or c[0] == 'delete' for c in calls))


if __name__ == '__main__':
    unittest.main()
