###############################################################################
#                                                                             #
#  Copyright (C) 2018 Teodor-Stelian Ionescu <teoionescu32@gmail.com>         #
#                                                                             #
#  This file is part of BOSSLANG source to source interpreter.                #
#  Unauthorized copying of this file, via any medium is warmly encouraged.    #
#                                                                             #
###############################################################################

from collections import OrderedDict
from stdmethods import *

###############################################################################
#                                                                             #
#  BOSSLANG LEXER                                                             #
#                                                                             #
###############################################################################

INTEGER       = 'INTEGER'
REAL          = 'REAL'
INTEGER_CONST = 'INTEGER_CONST'
REAL_CONST    = 'REAL_CONST'
PLUS          = 'PLUS'
MINUS         = 'MINUS'
MUL           = 'MUL'
INTEGER_DIV   = 'INTEGER_DIV'
FLOAT_DIV     = 'FLOAT_DIV'
LPAREN        = 'LPAREN'
RPAREN        = 'RPAREN'
SQLPAREN      = 'SQLPAREN' #
SQRPAREN      = 'SQRPAREN' #
ID            = 'ID'
ASSIGN        = 'ASSIGN'
BEGIN         = 'BEGIN'
END           = 'END'
SEMI          = 'SEMI'
DOT           = 'DOT'
PROGRAM       = 'PROGRAM'
VAR           = 'VAR'
COLON         = 'COLON'
COMMA         = 'COMMA'
EOF           = 'EOF'
COMP_EQ       = 'COMP_EQ' #
COMP_NEQ      = 'COMP_NEQ' #
COMP_S        = 'COMP_S' #
COMP_L        = 'COMP_L' #
COMP_SEQ      = 'COMP_SEQ' #
COMP_LEQ      = 'COMP_LEQ' #
IF            = 'IF' #
WHILE         = 'WHILE' #
THEN          = 'THEN' #
ELSE          = 'ELSE' #
CALL          = 'CALL' #

class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def __str__(self):
        return 'Token({type}, {value})'.format(type=self.type, value=repr(self.value))
    def __repr__(self):
        return self.__str__()

RESERVED_KEYWORDS = {
    'PROGRAM': Token('PROGRAM', 'PROGRAM'),
    'VAR': Token('VAR', 'VAR'),
    'DIV': Token('INTEGER_DIV', 'DIV'),
    'INTEGER': Token('INTEGER', 'INTEGER'),
    'REAL': Token('REAL', 'REAL'),
    'BEGIN': Token('BEGIN', 'BEGIN'),
    'END': Token('END', 'END'),
    'IF': Token('IF', 'IF'),
    'WHILE': Token('WHILE', 'WHILE'),
    'THEN': Token('THEN', 'THEN'),
    'ELSE': Token('ELSE', 'ELSE'),
    'CALL': Token('CALL', 'CALL'),
}

class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_line = 1
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('line {} -> Invalid character'.format(self.current_line))

    def advance(self):
        if self.current_char == '\n':
            self.current_line += 1
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while (
                self.current_char is not None and
                self.current_char.isdigit()
            ):
                result += self.current_char
                self.advance()

            token = Token('REAL_CONST', float(result))
        else:
            token = Token('INTEGER_CONST', int(result))

        return token

    def _id(self):
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result.upper(), Token(ID, result))
        return token

    def check_token(self, op):
        if len(op) == 2:
            if self.current_char == op[0] and self.peek() == op[1]:
                self.advance()
                self.advance()
                return True
        if len(op) == 1:
            if self.current_char == op[0]:
                self.advance()
                return True
        return False

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.advance()
                while self.current_char != '}':
                    self.advance()
                self.advance()
                continue

            if self.current_char.isalpha():
                return self._id()
            if self.current_char.isdigit():
                return self.number()

            if self.check_token(':='):
                return Token(ASSIGN, ':=')
            if self.check_token(';'):
                return Token(SEMI, ';')
            if self.check_token(':'):
                return Token(COLON, ':')
            if self.check_token(','):
                return Token(COMMA, ',')
            if self.check_token('+'):
                return Token(PLUS, '+')
            if self.check_token('-'):
                return Token(MINUS, '-')
            if self.check_token('*'):
                return Token(MUL, '*')
            if self.check_token('/'):
                return Token(FLOAT_DIV, '/')
            if self.check_token('='):
                return Token(COMP_EQ, '=')
            if self.check_token('<>'):
                return Token(COMP_NEQ, '<>')
            if self.check_token('<='):
                return Token(COMP_SEQ, '<=')
            if self.check_token('<'):
                return Token(COMP_S, '<')
            if self.check_token('>='):
                return Token(COMP_LEQ, '>=')
            if self.check_token('>'):
                return Token(COMP_L, '>')
            if self.check_token('('):
                return Token(LPAREN, '(')
            if self.check_token(')'):
                return Token(RPAREN, ')')
            if self.check_token('['):
                return Token(SQLPAREN, '[')
            if self.check_token(']'):
                return Token(SQRPAREN, ']')
            if self.check_token('.'):
                return Token(DOT, '.')
            self.error()

        return Token(EOF, None)

###############################################################################
#                                                                             #
#  BOSSLANG PARSER                                                            #
#                                                                             #
###############################################################################
class AST(object):
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr

class Compound(AST):
    def __init__(self):
        self.children = []

class Conditional(AST):
    def __init__(self, cond, op, then, elseif):
        self.cond = cond
        self.token = self.op = op
        self.then = then
        self.elseif = elseif

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Var(AST):
    def __init__(self, token, indicies = []):
        self.token = token
        self.value = token.value
        self.indicies = indicies

class NoOp(AST):
    pass

class Program(AST):
    def __init__(self, name, block):
        self.name = name
        self.block = block

class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement

class Func(AST):
    def __init__(self, function_name, argument_list, should_normalize):
        self.function_name = function_name
        self.argument_list = argument_list
        self.should_normalize = should_normalize

class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('line {} -> Invalid syntax'.format(self.lexer.current_line))

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        """1 program : PROGRAM variable SEMI block DOT"""
        self.eat(PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(SEMI)
        block_node = self.block()
        program_node = Program(prog_name, block_node)
        self.eat(DOT)
        return program_node

    def block(self):
        """2 block : declarations compound_statement"""
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        node = Block(declaration_nodes, compound_statement_node)
        return node

    def declarations(self):
        """3 declarations : VAR (variable_declaration SEMI)+ | empty"""
        declarations = []
        while True:
            if self.current_token.type == VAR:
                self.eat(VAR)
                while self.current_token.type == ID:
                    var_decl = self.variable_declaration()
                    declarations.extend(var_decl)
                    self.eat(SEMI)
            else:
                break
        return declarations

    def variable_declaration(self):
        """4 variable_declaration : ID (COMMA ID)* COLON type_spec"""
        var_nodes = [Var(self.current_token)]
        self.eat(ID)

        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(ID)

        self.eat(COLON)

        type_node = self.type_spec()
        var_declarations = [
            VarDecl(var_node, type_node)
            for var_node in var_nodes
        ]
        return var_declarations

    def type_spec(self):
        """5 type_spec : INTEGER | REAL"""
        token = self.current_token
        if self.current_token.type == INTEGER:
            self.eat(INTEGER)
        else:
            self.eat(REAL)
        node = Type(token)
        return node

    def compound_statement(self):
        """6 compound_statement : BEGIN statement_list END"""
        self.eat(BEGIN)
        nodes = self.statement_list()
        self.eat(END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        """7 statement_list : statement | statement SEMI statement_list"""
        node = self.statement()
        results = [node]

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            results.append(self.statement())

        return results

    def statement(self):
        """8 statement : compound_statement | assignment_statement | conditional_statement | funcall | empty"""
        if self.current_token.type == BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == ID:
            node = self.assignment_statement()
        elif self.current_token.type == IF:
            node = self.conditional_statement()
        elif self.current_token.type == WHILE:
            node = self.conditional_statement()
        elif self.current_token.type == CALL:
            node = self.funcall()
        else:
            node = self.empty()
        return node

    def arguments(self):
        """8''' arguments : (COMMA comexpr)*"""
        args = []
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            args.append(self.comexpr())
        return args
    
    def funcall(self, should_normalize = False):
        """8'' funcall : CALL LPAREN ID arguments RPAREN"""
        self.eat(CALL)
        self.eat(LPAREN)
        function_name = self.current_token.value
        self.eat(ID)
        argument_list = self.arguments()
        self.eat(RPAREN)
        node = Func(function_name, argument_list, should_normalize)
        return node
    
    def conditional_statement(self):
        """8' conditional_statement : IF comexpr THEN statement (ELSE statement)? | WHILE comexpr THEN statement"""
        token = self.current_token
        if self.current_token.type == IF:
            self.eat(IF)
        elif self.current_token.type == WHILE:
            self.eat(WHILE)
        cond = self.comexpr()
        self.eat(THEN)
        then = self.statement()
        elseif = self.empty()
        if token.type == IF and self.current_token.type == ELSE:
            self.eat(ELSE)
            elseif = self.statement()
        node = Conditional(cond, token, then, elseif)
        return node

    def assignment_statement(self):
        """9 assignment_statement : variable ASSIGN comexpr"""
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.comexpr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        """10 variable : ID (SQLPAREN comexpr SQRPAREN)? (SQLPAREN comexpr SQRPAREN)?"""
        var_name = self.current_token
        self.eat(ID)
        indicies = []
        while self.current_token.type == SQLPAREN:
            self.eat(SQLPAREN)
            indicies.append(self.comexpr())
            self.eat(SQRPAREN)
        return Var(var_name, indicies)

    def empty(self):
        """11 empty : _"""
        return NoOp()

    def comexpr(self):
        """12' comexpr : expr ((COMP_EQ | COMP_NEQ | COMP_S | COMP_L | COMP_SEQ | COMP_LEQ) expr)*"""
        node = self.expr()
        while self.current_token.type in (COMP_EQ, COMP_NEQ, COMP_S, COMP_L, COMP_SEQ, COMP_LEQ):
            token = self.current_token
            if token.type == COMP_EQ:
                self.eat(COMP_EQ)
            elif token.type == COMP_NEQ:
                self.eat(COMP_NEQ)
            elif token.type == COMP_S:
                self.eat(COMP_S)
            elif token.type == COMP_L:
                self.eat(COMP_L)
            elif token.type == COMP_SEQ:
                self.eat(COMP_SEQ)
            elif token.type == COMP_LEQ:
                self.eat(COMP_LEQ)
            node = BinOp(left=node, op=token, right=self.expr())
        return node

    def expr(self):
        """12 expr : term ((PLUS | MINUS) term)*"""
        node = self.term()
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            node = BinOp(left=node, op=token, right=self.term())
        return node 

    def term(self):
        """13 term : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*"""
        node = self.factor()
        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif token.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)
            node = BinOp(left=node, op=token, right=self.factor())
        return node

    def factor(self):
        """14 factor : PLUS factor | MINUS factor | INTEGER_CONST | REAL_CONST | LPAREN comexpr RPAREN | funcall | variable"""
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.comexpr()
            self.eat(RPAREN)
            return node
        elif token.type == CALL:
            node = self.funcall(True)
            return node
        else:
            node = self.variable()
            return node

    def parse(self):
        """
        1 program
        2 block
        3 declarations
        4 variable_declaration
        5 type_spec
        6 compound_statement
        7 statement_list
        8 statement
        8''' arguments
        8'' funcall
        8' conditional_statement
        9 assignment_statement
        10 variable
        11 empty
        12' comexpr
        12 expr
        13 term
        14 factor
        """
        node = self.program()
        if self.current_token.type != EOF:
            self.error()
        return node

###############################################################################
#                                                                             #
#  BOSSLANG AST reflection walker                                             #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

###############################################################################
#                                                                             #
#  BOSSLANG SYMBOLS, TABLES, SEMANTIC ANALYSIS                                #
#                                                                             #
###############################################################################

class Symbol(object):
    def __init__(self, name, type=None):
        self.name = name
        self.type = type

class VarSymbol(Symbol):
    def __init__(self, name, type):
        super(VarSymbol, self).__init__(name, type)
    def __str__(self):
        return "<{class_name}(name='{name}', type='{type}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            type=self.type,)
    __repr__ = __str__

class BuiltinTypeSymbol(Symbol):
    def __init__(self, name):
        super(BuiltinTypeSymbol, self).__init__(name)
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<{class_name}(name='{name}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,)

class SymbolTable(object):
    def __init__(self):
        self._symbols = OrderedDict()
        self._init_builtin_types()
    def __str__(self):
        symtab_header = 'Symbol table contents'
        lines = ['\n', symtab_header, '_' * len(symtab_header)]
        lines.extend(
            ('%7s: %r' % (key, value))
            for key, value in self._symbols.items()
        )
        lines.append('\n')
        s = '\n'.join(lines)
        return s
    __repr__ = __str__

    def _init_builtin_types(self):
        self.insert(BuiltinTypeSymbol('INTEGER'))
        self.insert(BuiltinTypeSymbol('REAL'))

    def insert(self, symbol):
        self._symbols[symbol.name] = symbol

    def lookup(self, name):
        symbol = self._symbols.get(name)
        # <Maybe None>
        return symbol

class SemanticAnalyzer(NodeVisitor):
    def __init__(self):
        self.symtab = SymbolTable()

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Num(self, node):
        pass
    
    def visit_NoOp(self, node):
        pass
    
    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_VarDecl(self, node):
        type_name = node.type_node.value
        type_symbol = self.symtab.lookup(type_name)

        var_name = node.var_node.value
        var_symbol = VarSymbol(var_name, type_symbol)

        if self.symtab.lookup(var_name) is not None:
            raise Exception("Error: Duplicate identifier '%s' found" % var_name)

        self.symtab.insert(var_symbol)

    def visit_Conditional(self, node):
        self.visit(node.cond)
        self.visit(node.then)
        self.visit(node.elseif)

    def visit_Assign(self, node):
        self.visit(node.right)
        self.visit(node.left)

    def visit_Func_Checker(self, name, numargs):
        try:
            exec("exec_return = hasattr(%s, '__call__')" % name, globals())
            if exec_return == True:
                try:
                    import sys
                    if sys.version_info[0] == 2:
                        exec("import inspect; exec_return = inspect.getargspec(%s)" % name, globals())
                        count = len(exec_return.args) - len(exec_return.defaults)
                    elif sys.version_info[0] == 3:
                        exec("import inspect; exec_return = inspect.signature(%s)" % name, globals())
                        count = len([x for x in exec_return.parameters.values() if x.default == x.empty])
                    if numargs != count:
                        return "Error: Function '{}' arguments expected {}, found {}".format(name, count, numargs)
                    else:
                        return ""
                except TypeError as e:
                    # python2 fails __builtin__ function
                    return ""
                except ValueError as e:
                    # python3 fails __builtin__ function
                    return ""
            else:
                return "Error: '{}' is not a function".format(name)
        except NameError as e:
            return "Error: Function name '{}' is not defined".format(name)
        except SyntaxError as e:
            return ""
    
    def visit_Func(self, node):
        verdict = self.visit_Func_Checker(node.function_name, len(node.argument_list))
        if verdict:
            raise Exception(verdict)
        for argument in node.argument_list:
            self.visit(argument)

    def visit_Var(self, node):
        var_name = node.value
        var_symbol = self.symtab.lookup(var_name)
        if len(node.indicies) > 2:
            raise Exception("Error: Too many array indexing operators on identifier '%s'" % var_name)
        for indexer in node.indicies:
            self.visit(indexer)
        if var_symbol is None:
            raise Exception("Error: Symbol(identifier) not found '%s'" % var_name)
    
    def visit_NoneType(self, node):
        import traceback
        traceback.print_stack()
        raise Exception("Fatal error occured")

###############################################################################
#                                                                             #
#  BOSSLANG PYTHON INTERPRETER                                                #
#                                                                             #
###############################################################################

class Interpreter(NodeVisitor):
    def __init__(self, tree):
        self.tree = tree

    def visit_Program(self, node):
        return self.visit(node.block)

    def visit_Block(self, node):
        lines = []
        for declaration in node.declarations:
            lines.append(self.visit(declaration))
        lines.append(self.visit(node.compound_statement))
        return '\n'.join(lines)

    def visit_VarDecl(self, node):
        expanded_indicies = []
        for index in node.var_node.indicies:
            expanded_indicies.append(self.visit(index))
        return "_ASSIGN('{}', {}, {})".format(
            node.var_node.value, 
            '[' + ', '.join(expanded_indicies) + ']',
            self.visit(node.type_node),
        )

    def visit_Type(self, node):
        if node.value == INTEGER:
            return "0"
        elif node.value == REAL:
            return "float(0)"

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return "({}) + ({})".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == MINUS:
            return "({}) - ({})".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == MUL:
            return "({}) * ({})".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == INTEGER_DIV:
            return "({}) // ({})".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == FLOAT_DIV:
            return "(float({})) / (float({}))".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == COMP_EQ:
            return "int(({}) == ({}))".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == COMP_NEQ:
            return "int(({}) != ({}))".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == COMP_S:
            return "int(({}) < ({}))".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == COMP_L:
            return "int(({}) > ({}))".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == COMP_SEQ:
            return "int(({}) <= ({}))".format(self.visit(node.left), self.visit(node.right))
        elif node.op.type == COMP_LEQ:
            return "int(({}) >= ({}))".format(self.visit(node.left), self.visit(node.right))

    def visit_Num(self, node):
        return str(node.value)

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return "+({})".format(self.visit(node.expr))
        elif op == MINUS:
            return "-({})".format(self.visit(node.expr))

    def visit_Compound(self, node):
        lines = []
        for child in node.children:
            lines.append(self.visit(child))
        lines = [x for x in lines if x]
        return '\n'.join(lines)
    
    def visit_Conditional(self, node):
        block = '\n' + '\n'.join(map(lambda c: '\t' + c, self.visit(node.then).split('\n')))
        if len(block.split()) == 0:
            block = "\n\tpass"
        if node.op.type == IF:
            else_block = '\n' + '\n'.join(map(lambda c: '\t' + c, self.visit(node.elseif).split('\n')))
            if len(else_block.split()) == 0:
                else_block = "\n\tpass"
            return "if _CONDITION({}):{}\nelse:{}".format(self.visit(node.cond), block, else_block)
        elif node.op.type == WHILE:
            return "while _CONDITION({}):{}".format(self.visit(node.cond), block)
    
    def visit_Assign(self, node):
        var_name = node.left.value
        var_value = self.visit(node.right)
        expanded_indicies = []
        for index in node.left.indicies:
            expanded_indicies.append(self.visit(index))
        return "_ASSIGN('{}', {}, {})".format(
            var_name,
            '[' + ', '.join(expanded_indicies) + ']',
            var_value
        )
    
    def visit_Func(self, node):
        args = []
        for argument in node.argument_list:
            args.append(self.visit(argument))
        func_block = "{}({})".format(node.function_name, ', '.join(args))
        if node.should_normalize:
            func_block = "_NORMALIZE({})".format(func_block)
        return func_block

    def visit_Var(self, node):
        expanded_indicies = []
        for index in node.indicies:
            expanded_indicies.append(self.visit(index))
        return "_GET('{}', {})".format(node.value, '[' + ', '.join(expanded_indicies) + ']')

    def visit_NoOp(self, node):
        return ""

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)

if __name__ == '__main__':
    pass
