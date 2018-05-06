
from collections import OrderedDict
import sys
sys.path.insert(0, './lib')
from stdmethods import *
from compiler import Lexer, Parser, SemanticAnalyzer, Interpreter

# Underlying runtime BOSSLANG python methods

def _NORMALIZE(value):
    if type(value) != int and type(value) != float:
        return 0
    return value

def _SUBSCRIPT(var_name, index):
    while len(index) < 2:
        index.append(0)
    if type(index[0]) is not int:
        raise Exception("Array '{}' subscript '{}' is not an integer".format(var_name, index[0]))
    if type(index[1]) is not int:
        raise Exception("Array '{}' subscript '{}' is not an integer".format(var_name, index[1]))
    return "[" + str(index[0]) + "][" + str(index[1]) + "]"

def _ASSIGN(var_name, index, var_value):
    GLOBAL_MEMORY[var_name + _SUBSCRIPT(var_name, index)] = var_value

def _GET(var_name, index):
    return GLOBAL_MEMORY[var_name + _SUBSCRIPT(var_name, index)]

def _CONDITION(value):
    return value != 0

# __main__

def _runsource(source):
    try:
        lexer = Lexer(source)
        parser = Parser(lexer)
        tree = parser.parse()
        semantic_analyzer = SemanticAnalyzer()
        semantic_analyzer.visit(tree)
    except Exception as e:
        print(e)
        return

    # print(semantic_analyzer.symtab)
    interpreter = Interpreter(tree)
    result = interpreter.interpret()
    if not result:
        result = "pass"
    transpiled_source = "def USER_PROGRAM():\n" + '\n'.join(map(lambda c: '\t' + c, result.split('\n'))) + \
        "\nif __name__ == '__main__':\n\tGLOBAL_MEMORY = OrderedDict()\n\tUSER_PROGRAM()"

    # print(transpiled_source)
    codeobj = compile(transpiled_source, 'submodule', 'exec')
    try:
        exec(codeobj, globals())
    except Exception as e:
        print("Runtime exception of type {} occurred:\n{}".format(type(e).__name__, \
        '\n'.join(map(lambda c: '>\t' + c, str(e).split('\n')))))

def _main():
    import argparse
    parser = argparse.ArgumentParser(description='Bosslang compiler.')
    parser.add_argument('source', metavar='inputfile', type=str, help='source file to compile')
    args = parser.parse_args()
    try:
        text = open(args.source, 'r').read()
        _runsource(text)
    except Exception as e:
        print(e)

# EXPORT

def runsource(source):
    _runsource(source)

if __name__ == '__main__':
    _main()
