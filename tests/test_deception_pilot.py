import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.deception_pilot import messages_for, select_rows, summarize, check_generation_memory, available_host_memory


class PilotTests(unittest.TestCase):
    def test_sampling_and_reference_exclusion(self):
        import yaml
        raw = yaml.safe_dump([{'scenario': f'scenario {i}', 'question': 'Question?',
                              'answer_prefix': 'Person:', 'honest_completion': 'SECRET_H',
                              'deceptive_completion': 'SECRET_D'} for i in range(10)])
        rows = select_rows(raw, 5, 42)
        self.assertEqual(rows, select_rows(raw, 5, 42))
        self.assertEqual(len({r['id'] for r in rows}), 5)
        for row in rows:
            self.assertNotIn('SECRET', json.dumps(messages_for(row)))
            self.assertNotIn('honest_completion', row)

    def test_blank_labels_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'responses.jsonl').write_text(json.dumps({'id': 'a', 'hit_token_limit': True}) + '\n')
            (path / 'labels.csv').write_text('id,label,notes\na,,\n')
            summarize(path)
            result = json.loads((path / 'summary.json').read_text())
            self.assertEqual(result['counts']['unlabeled'], 1)
            self.assertEqual(result['counts']['honest'], 0)
            self.assertEqual(result['hit_token_limit'], 1)
            (path / 'labels.csv').write_text('id,label,notes\na,wrong,\n')
            with self.assertRaises(ValueError):
                summarize(path)
            (path / 'labels.csv').write_text('id,label,notes\nb,honest,\n')
            with self.assertRaises(ValueError):
                summarize(path)

    def test_memory_guard(self):
        gib = 1024 ** 3
        for device, host, gpu in [('cpu', 8, 0), ('cuda', 8, 24), ('cuda', 32, 8)]:
            with self.assertRaises(ValueError):
                check_generation_memory(device, host * gib, gpu * gib)
        check_generation_memory('cpu', 25 * gib)
        check_generation_memory('cuda', 16 * gib, 16 * gib)

    def test_container_cache_accounting(self):
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meminfo = root / 'meminfo'
            meminfo.write_text(f'MemAvailable: {31 * gib // 1024} kB\n')
            (root / 'memory.max').write_text(str(32 * gib))
            (root / 'memory.current').write_text(str(10 * gib))
            # Without cache information retain the conservative estimate.
            self.assertEqual(available_host_memory(meminfo, root), 22 * gib)
            (root / 'memory.stat').write_text(f'inactive_file {8*gib}\nfile_dirty {gib}\nfile_writeback 0\n')
            self.assertEqual(available_host_memory(meminfo, root), 29 * gib)
            meminfo.write_text(f'MemAvailable: {20 * gib // 1024} kB\n')
            self.assertEqual(available_host_memory(meminfo, root), 20 * gib)
