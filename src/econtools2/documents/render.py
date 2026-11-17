import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.documents.latex import Multipart, MCQ, RawLatex


def get_online_format(problem, setup_id=None, question_ids=None):
    setup = problem.setup_list[setup_id]['online_setup']
    if len(setup)>0:
        setup = '<p>'+setup+'</p>\n'
    else:
        setup = ''
    solution = ''
    i=1
    for qid in question_ids:
        setup+=f'<p>{i}. ' + problem.question_list[qid]['online_question'] + '</p>\n'
        solution+=f'<p>{i}. ' + problem.question_list[qid]['online_answer'] + '</p>\n'
        i+=1
    return {'setup': setup, 'solution': solution}


def get_multipart_sa(problem, setup_id=None, question_ids=None):
    setup = problem.setup_list[setup_id]['setup']
    assert len(setup)>0
    t = "\\begin{q}\n"
    t+= setup
    t+= "\\begin{enumerate}[a.]\n"
    for qid in question_ids:
        t+="\\item " + problem.question_list[qid]['question'] + "\n"
    t+= "\\end{enumerate}\n"
    t+= "\\end{q}\n"
    return RawLatex(t)


def get_multipart_mcq(problem, setup_id=None, question_ids=None):
    multipart = Multipart()
    setup = problem.setup_list[setup_id]['setup']
    assert len(setup)>0
    multipart.add(RawLatex(setup))
    for qid in question_ids:
        multipart.add(problem.question_list[qid]['MCQ'])
    return multipart


def get_single_sa(problem, setup_id=None, question_id=None):
    setup = problem.setup_list[setup_id]['setup']
    question = problem.question_list[question_id]['question']
    if len(setup)>0:
        question = setup + "\n\n" + question
    t = "\\begin{q}\n"
    t+= question
    t+= "\\end{q}\n"
    return RawLatex(t)


def get_single_mcq(problem, setup_id=None, question_id=None):
    setup = problem.setup_list[setup_id]['setup']
    mcq = problem.question_list[question_id]['MCQ']
    if len(setup)>0:
        mcq.question = setup + "\n\n" + mcq.question
    return mcq
