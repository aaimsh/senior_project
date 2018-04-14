import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from flaskr.util import train_test as tg

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
    # DATABASE=os.path.join(app.root_path, 'flaskr.db'),
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team/')
def team():
    return render_template('team.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/oktop/')
def oktop():
    result = tg.oktop()
    return render_template("result.html", result=result)
