#coding:utf-8
# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import os
import sys

def _load_modules(root: str):
    """ 
    Dynamically load modules from all the files ending with '.py' under current root and return a list. 
    """
    
    modules = []
    for filename in os.listdir(root):
        if filename.endswith(".py"):
            name = os.path.splitext(filename)[0]
            if name.isidentifier():
                fh = None
                try:
                    fh = open(filename, "r", encoding="utf8")
                    code = fh.read()
                    module = type(sys)(name)
                    sys.modules[name] = module
                    exec(code, module.__dict__)
                    modules.append(module)
                except (EnvironmentError, SyntaxError) as err:
                    sys.modules.pop(name, None)
                    print(err)
                finally:
                    if fh is not None:
                        fh.close()
    return modules