# run sphinx

import subprocess
import unittest
import os

def parse_return(subprocessreturn):
    if subprocessreturn == 0:
        return True
    else:
        raise RuntimeError('The sphinx build returned an error or warning. Run '
                           '"sphinx-build -b html ./docs ./docs/_build"'
                           ' from the root checkout folder and fix any warnings or errors.')

class DocBuildTestCase(unittest.TestCase):
    def test_build_sphinx_docs(self):
        rightplace = os.path.exists("./docs")
        if not rightplace:
            raise IOError('Docbuild test not starting in right folder. No /docs folder found')
        conffileexists = os.path.isfile("./docs/conf.py")
        if not conffileexists:
            raise IOError('Docbuild test not starting in right folder. No /docs/conf.py file found')
        result = subprocess.check_call(['sphinx-build','-W','-b','html','./docs','./docs/_build'])
        test = parse_return(result)
        self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()