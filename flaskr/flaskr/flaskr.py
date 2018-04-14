import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from .text_generator import Writer


writer = Writer()
writer.load("model_info/")
app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
    #DATABASE=os.path.join(app.root_path, 'flaskr.db'),
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
    result = writer.write(0, number_of_word=400)
    return render_template("result.html", result=result)

if __name__ == '__main__':
    app.run()