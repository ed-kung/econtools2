import os
import subprocess
from importlib.resources import files

import numpy as np

from econtools2._rng import default_rng

DEFAULT_PREAMBLE = files('econtools2.documents').joinpath('templates/preamble.tex').read_text()


class Document:
    def __init__(self, separator='\n\n'):
        self.items = []
        self.separator = separator
    
    def add(self, item):
        self.items.append(item)
    
    def texify(self, solutions=False):
        t = DEFAULT_PREAMBLE
        t+= "\\begin{document}\n\n"
        
        for item in self.items:
            t+="\\begin{minipage}{\\textwidth}\n"
            t+=item.texify(solutions)
            t+=self.separator
            t+="\\end{minipage}\n\n"
        
        t+="\\end{document}\n\n"
        return t
    
    def print_pdf(self, filename, output_dir, solutions=False):
        t = self.texify(solutions)
        with open(os.path.join(output_dir, f"{filename}.tex"), 'w') as f:
            f.write(t)
        curr_dir = os.getcwd()
        os.chdir(output_dir)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", filename],
            capture_output=True,
            text=True,
            timeout=60
        )
        print(result.stdout)
        print(result.stderr)
        os.chdir(curr_dir)


class Multipart:
    def __init__(self):
        self.items = []
    
    def add(self, item):
        self.items.append(item)
    
    def texify(self, solutions=False):
        t = ''
        for item in self.items:
            t+=item.texify(solutions)
        return t


class Homework(Document):
    def __init__(self, title, due_date=None, separator='\\vspace{0.5cm}\n\\hrule\n\\vspace{0.5cm}'):
        self.items = []
        self.separator=separator
        if due_date:
            self.items.append(RawLatex(f"\\noindent {title} \\\\ \\noindent Due: {due_date}"))
        else:
            self.items.append(RawLatex(f"\\noindent {title}"))


class Exam(Document):
    def __init__(self, title, version, time, separator='\\vspace{0.5cm}\n\\hrule\n\\vspace{0.5cm}'):
        self.items = []
        self.separator=separator
        self.items.append(RawLatex(f"\\noindent {title} \\\\ \\noindent Version: {version} \\\\ \\noindent Time: {time} minutes"))


class RawLatex:
    def __init__(self, tex):
        self.tex = tex
    
    def texify(self, solutions=False):
        return self.tex


class MCQ:
    def __init__(self, question, answers, solution=0, horz=False, shuffle=True, sort=False, rng=None, numerical=False):
        rng = default_rng(rng)
        self.question = question
        self.answers = answers
        self.solution = solution
        self.horz = horz
        self.numerical = numerical
        
        if sort:
            self.answers = answers.copy()
            sortindex = np.argsort(answers)
            for i in range(len(answers)):
                self.answers[i] = answers[sortindex[i]]
            self.solution = sortindex.tolist().index(solution)
        elif shuffle:
            perm = rng.permutation(len(self.answers))
            tempa = self.answers.copy()
            for i in range(len(self.answers)):
                self.answers[perm[i]] = tempa[i]
            self.solution = perm[self.solution]
        
        # Check duplicates
        if len(self.answers)!=len(set(self.answers)):
            print(self.answers)
            print("Error: Duplicate answers!")
            raise
    
    def texify(self, solutions=False):
        question = self.question
        answers = self.answers
        t = "\\begin{q}\n" + question + "\n\n"
        if self.horz:
            for i in range(len(answers)):
                myans = answers[i]
                if self.numerical:
                    myans = f"{myans:,g}"
                t+= f"~ {chr(97+i)}) {myans} \n"
        else:
            t+='\\begin{enumerate}[(a)]\n'
            for i in range(len(answers)):
                myans = answers[i]
                if self.numerical:
                    myans = f"{myans:,g}"
                t+=f"  \\item {myans} \n"
            t+='\\end{enumerate}\n'
        if solutions:
            t+= f"\n{{\\color{{red}} Answer: {chr(97+self.solution)} }}\n"
        t+= "\\end{q}\n"
        return t
