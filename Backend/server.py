from flask import Flask
from flask_cors import CORS  # Import CORS from flask_cors

from router import router

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # Enable CORS for the Flask app
# Register the router blueprint

app.register_blueprint(router)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
