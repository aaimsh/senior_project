import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify
from .text_generator import Writer, get_prediction


forward_writer= Writer()
forward_writer.load(path='model_info/', name='forward')
#*************note begin*************
#here sould be the initialization of backward_writer
# the code is:
#backward_writer= Writer()
#backward_writer.load(path='model_info/', name='backward')
#*************note end*************

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
            result = forward_writer.write(1, words=init.split(), number_of_word=num)
    else:
        result = forward_writer.write(0, number_of_word=400)
    return render_template('result.html', result=result)
#*************note begin*************
#here should be a method called (predict) and work like this:
#First the data should be taken from the user as list of string as in (oktop) method
#Second applaying this code:
#result = get_prediction(forward_writer, backward_writer, data)
#---> where (data) is user input and (result) is output
#*************note end*************
if __name__ == '__main__':
    app.run()