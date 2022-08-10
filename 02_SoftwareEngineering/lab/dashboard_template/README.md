# Web App Dashboard Template

This example comes originally from the Udacity repository [DSND_Term2](https://github.com/udacity/DSND_Term2).

It shows a very simple web app dashboard template in which Flask and Plotly are used to plot some simple interactive diagrams on a website.

The web app runs locally, no deployment is done.

To see how to deploy such apps, have a look at [data_science_udacity](https://github.com/mxagar/data_science_udacity):

`02_SoftwareEngineering / DSND_SWEngineering.md - 6.8 Deployment on Heroku`.

#### Files

```
./dashboard_template git:(main) ✗ tree
.
├── README.md
├── myapp
│   ├── __init__.py
│   ├── routes.py # Flask routes: we pass the Python back-end objects to the front-end
│   ├── static
│   │   └── img
│   │       ├── githublogo.png
│   │       └── linkedinlogo.png
│   └── templates
│       └── index.html # Dashboard HTML, the front-end
├── myapp.py # The Flask web app is instantiated
├── requirements.txt
└── wrangling_scripts
    └── wrangle_data.py # Data wrangling and Plotly objects

```

#### How to Use This

Install the `requirements.txt` in your environment:

```
click==8.1.3
Flask==2.1.0
gunicorn==20.1.0
importlib-metadata==4.12.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.1
numpy==1.21.6
pandas==1.2.4
plotly==5.3.1
python-dateutil==2.8.2
pytz==2022.1
six==1.16.0
tenacity==8.0.1
typing_extensions==4.3.0
Werkzeug==2.2.1
zipp==3.8.1
```

Then, execute the Flask server:

```bash
cd dashboard_template
python myapp.py
```

Finally, open the browser in the specified port and address: `localhost:3001/`