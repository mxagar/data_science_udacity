# This is a package init file, the first which is run,
# thus we instantiate the app.
# Additionally, we import and instantiate routes,
# which contains all the website rendering calls.
from flask import Flask

app = Flask(__name__)

from worldbankapp import routes