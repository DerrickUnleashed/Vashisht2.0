from flask import Flask
from flask_cors import CORS  
from router import router

app = Flask(__name__)

CORS(app)  

app.register_blueprint(router)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
