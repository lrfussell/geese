import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model.
with open(f'model/geese.pkl', 'rb') as f:
    model = pickle.load(f)
    
app = flask.Flask(__name__, template_folder='templates')

# Use pickle to load in the pre-trained model.
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        AGELVL = flask.request.form['AGELVL']
        EDLVL = flask.request.form['EDLVL']
        GSEGRD = flask.request.form['GSEGRD']
        PATCO = flask.request.form['PATCO']
        STEMOCC = flask.request.form['STEMOCC']
        LOS = flask.request.form['LOS']
        input_variables = pd.DataFrame([[AGELVL, EDLVL, GSEGRD, PATCO, STEMOCC, LOS]],
                                       columns=['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'AGELVL':AGELVL,
                                                     'EDLVL':EDLVL,
                                                     'GSEGRD':GSEGRD,
                                                     'PATCO':PATCO,
                                                     'STEMOCC':STEMOCC,
                                                     'LOS':LOS},
                                     result = prediction,
                                     )

if __name__ == '__main__':
    app.run()

