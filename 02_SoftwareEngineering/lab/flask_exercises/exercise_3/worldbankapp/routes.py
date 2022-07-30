from worldbankapp import app
import json, plotly
from flask import render_template
from wrangling_scripts.wrangle_data import return_figures

@app.route('/')
@app.route('/index')
def index():
    # This example is more sophisticated than the one below:
    # We create Plotly graph objects with an external function, encode them as JSON
    # and pass them from the back-end to the front-end.
    # In the HTML pageJavascript Plotly is used to render them.

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # Note that we pass two things:
    # - a list: ids = [figure-0, figure-1, figure2, figure-3]
    # - a JSON with the figure plotting information
    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)

@app.route('/test')
def test_page():
    # This simple page view how a python object can be passed
    # from the back-end to the front-end:
    # Just pass it with any variable name to render_template()
    # and access that variable name with Jinja in the HTML file
    data = [1, 2, 3, 4, 5]
    return render_template('test.html', data=data)
