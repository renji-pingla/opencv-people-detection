from flask import Flask
from detector import getDetector
from detector import closeDector
app = Flask(__name__)

@app.route("/")
def index():
   return 'People Detection',getDetector()
   
@app.route('/close')
def hello():
    return 'Camera Closed',closeDector() 

if __name__ == '__main__':
    app.run(debug=True)
