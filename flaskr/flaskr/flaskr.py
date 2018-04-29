import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify
from .text_generator import Writer, get_prediction


forward_writer= Writer()
forward_writer.load(path='model_info/', name='forward')

#here sould be the initialization of backward_writer
backward_writer= Writer()
backward_writer.load(path='model_info/', name='backward')

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

@app.route('/oktop/', methods=['POST', ])
def oktop():
    if request.form['words'] != '':
        init = request.form['words']
        if int(request.form['num']) > 0 :
            num = int(request.form['num'])
            if request.form['mode'] == 'w':
                result = forward_writer.write(1, words=init.split(), number_of_word=num)
            elif request.form['mode'] == 'c':
                result = get_prediction(forward_writer, backward_writer, init.split())                
    elif request.form['mode'] == 'w':
        result = forward_writer.write(0, number_of_word=400)
    else:
        result = 'لا يوجد اي نص'
    return render_template('result.html', result=result)
if __name__ == '__main__':
    app.run()