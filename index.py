from flask import Flask, render_template, send_file
from detector import openCamera, removeCsv

app = Flask(__name__)
 




@app.route("/open")
def index():
   return 'People Detection',openCamera()   



@app.route("/get-csv-record")
def getRecord():
   return send_file('./people_records.csv',as_attachment=True)



@app.route("/delete")
def deleteCsv():
   return 'deleted', removeCsv()





if __name__ == '__main__':
    app.run(debug=True)
