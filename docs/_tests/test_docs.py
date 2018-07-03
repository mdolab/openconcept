# run sphinx

import subprocess
import unittest
import os

class DocBuildTestCase(unittest.TestCase):
    def test_build_sphinx_docs(self):
        rightplace = os.path.exists("./docs")
        if not rightplace:
            raise IOError('Docbuild test not starting in right folder. No /docs folder found')
        conffileexists = os.path.isfile("./docs/conf.py")
        if not conffileexists:
            raise IOError('Docbuild test not starting in right folder. No /docs/conf.py file found')
        result = subprocess.check_call(['sphinx-build','-M','html','./docs','./docs/_build'])
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()