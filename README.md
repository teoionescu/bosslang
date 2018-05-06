# bosslang
A basic Python source-to-source compiler and interpreter for a Pascal-like language.

## Usage

```bash
$ python exe.py source.boss
```
```bash
$ python -c "from exe import runsource; runsource('program test; begin end.')"
```

#### Description
`exe.py` breaks down the *Bosslang* source into an [*Abstract Syntax Tree (AST)*](http://en.wikipedia.org/wiki/Abstract_syntax_tree), reinterprets the output as a syntactically valid *Python* source and executes it.

## Syntax

### Program structure

```pascal
PROGRAM NONAME00;
    VAR a : INTEGER;
BEGIN
    { just a comment }
    a := 1;
END.
```
*Note: Reserved keywords are non-case-sensitive*\
*Note: Variable and function names are alphameric*

### Supported types

* `int()` and `float()` only
```pascal
var x, y : integer;
var r : real;
```
* Vectors and matrices also supported!
```pascal
program matrix;
var mat : integer;
begin
    mat[0][0] := 1;
    mat[0][1] := 2;
    mat[1=1][1=0] := 3;
    mat[CALL(max, 0, 1)][CALL(pow, 1, 1)] := 4;
    CALL(print, mat[1]); { prints mat[1][0] -> 3 }
end.
```
*Note: arrays and matrices have an indefinite size*\
*Note: strings not supported*\
*Note: type checking is ommited*

### Language constructs

- `if` statements
```pascal
if r <> 0 then begin
    if x = 0 then x := 1
    else x := -1;
end;
```
- `while` loops
```pascal
while x > 0 then begin
    CALL(print, x);
    x := x - 1;
end;
```

### Operator precendence

| Precedence | Operator | Description |
|:---:|:---:|:---:|
| 1 | \[ ] | Array subscripting |
| 2 | + -  | Unary plus/minus |
| 3 | * / DIV | Multiplication, division, integer division |
| 4 | + - | Binary plus/minus |
| 5 | = <> < <= > >=  | Relational operators|
| 6 | := | Assignment |

*Todo: add logical expressions*

### Functions

```pascal
returnval := CALL(<function_name> [, args...]);
```
* Any __builtin__ python or compiler-defined function is callable.

*Note: user-defined functions not supported*\
*Note: type incompatible functions return int(0)*
