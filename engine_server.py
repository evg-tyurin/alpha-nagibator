from flask import Flask, request
import os

from checkers.engine import *

""" 
    Flask-based server for running Kallisto API checkers engines which require Windows 32 bit environment. 
    Engine is loaded from DLL. DLL name is currently hard-coded in the init() method (see below).
    
    Engine does not support multiple matches at the same time.
    
    Note! If you don't use Kallisto engine(s) then you are NOT required to be on Windows 32 bit. 

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 19, 2018.

"""

app = Flask(__name__)
app.config['ENGINE'] = "None"

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/init', methods=['GET'])           
def init():
    """ Initializes an engine.
    """
    if app.config['ENGINE']=="None":
        app.config['ENGINE'] = KallistoApiEngine("./checkers/engine/KestoG_1_4_Moscow.dll")
        return "OK, "+app.config['ENGINE'].EI_GetName()
    else:
        return "OK, already initialized "+app.config['ENGINE'].EI_GetName()

@app.route('/reset', methods=['GET'])
def reset():
    if check_engine():
        app.config['ENGINE'].reset()
        return "OK"
    else:
        return "Engine not initialized"

@app.route('/think', methods=['GET'])
def think():
    if check_engine():
        return app.config['ENGINE'].think()
    else:
        return "Engine not initialized"

@app.route('/make_move', methods=['GET'])
def make_move():
    # make_move?list=g3:e5,e5:c3
    if check_engine():
        list = request.args.get('list')
        print("make_move?list="+list)
        for move in list.split(","):
            app.config['ENGINE'].makeMove(move)
        return "OK"
    else:
        return "Engine not initialized"

def check_engine():
    return app.config['ENGINE'] != "None"


if __name__ == "__main__":
    app.run("0.0.0.0", 8989)
    
