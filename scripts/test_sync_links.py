"""Offline regression checks for the bilingual link renderer."""
import copy
import json
import unittest

from sync_links import ROOT, render, validate


class LinkSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / 'data/paper_links.json').read_text())

    def test_catalog_and_bilingual_idempotence(self):
        validate(self.catalog)
        for name in ('README.md', 'README_ZH.md'):
            first = render((ROOT / name).read_text(), self.catalog)
            self.assertEqual(first, render(first, self.catalog))

    def test_missing_and_duplicate_rows_fail(self):
        catalog = copy.deepcopy(self.catalog)
        catalog['rows'].append(catalog['rows'][0])
        with self.assertRaises(ValueError):
            validate(catalog)
        with self.assertRaises(ValueError):
            render('', self.catalog)

    def test_table_shapes(self):
        for name in ('README.md', 'README_ZH.md'):
            count = None
            for line in render((ROOT / name).read_text(), self.catalog).splitlines():
                if not line.startswith('|'):
                    count = None
                    continue
                cells = line.strip('|').split('|')
                if count is None:
                    count = len(cells)
                self.assertEqual(count, len(cells), line)

    def test_changed_and_removed_urls_do_not_accumulate(self):
        catalog = copy.deepcopy(self.catalog)
        for paper in catalog['papers'].values():
            paper['paper'] = 'https://example.org/paper'
            paper['code'] = None
        text = render((ROOT / 'README.md').read_text(), self.catalog)
        updated = render(text, catalog)
        self.assertEqual(updated, render(updated, catalog))
        self.assertNotRegex(updated, r'\[Code\]\(https://github.com/')

    def test_announced_projects_and_invalid_urls(self):
        catalog = copy.deepcopy(self.catalog)
        for paper in catalog['papers'].values():
            paper['code_status'] = 'announced'
        text = render((ROOT / 'README.md').read_text(), catalog)
        self.assertIn('[Project](', text)
        self.assertNotIn('[Code](', text)
        self.assertEqual(text, render(text, catalog))
        next(iter(catalog['papers'].values()))['paper'] = 'javascript:alert(1)'
        with self.assertRaises(ValueError):
            validate(catalog)


if __name__ == '__main__':
    unittest.main()
