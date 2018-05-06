import unittest
import subprocess

class Matrix(unittest.TestCase):

    def ok(self):
        return "3\n"

    def test_python2(self):
        bashCommand = "python exe.py matrix.boss"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.assertTrue(output == self.ok() and error == None)
    
    def test_python3(self):
        bashCommand = "python3 exe.py matrix.boss"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.assertTrue(output == self.ok() and error == None)

class Unit(unittest.TestCase):

    def ok(self):
        return "372.36\n"
    
    def test_python2(self):
        bashCommand = "python exe.py unit.boss"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.assertTrue(output == self.ok() and error == None)
    
    def test_python3(self):
        bashCommand = "python3 exe.py unit.boss"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.assertTrue(output == self.ok() and error == None)

class Example(unittest.TestCase):

    def ok(self):
        return "6\n5\n4\n3\n2\n1\nExecute a move up command\n30\n900\n"

    def test_python2(self):
        bashCommand = "python exe.py example.boss"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.assertTrue(output == self.ok() and error == None)
    
    def test_python3(self):
        bashCommand = "python3 exe.py example.boss"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.assertTrue(output == self.ok() and error == None)

if __name__ == '__main__':
    unittest.main()