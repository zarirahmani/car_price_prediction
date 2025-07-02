
#I want to write a function to create a web servie and I use Flask for that.
from flask import Flask

app = Flask('ping')

#This will allow to turn this file into a web service that can be run independently.
@app.route('/ping', methods=['GET'])  # I want to access this function using GET method and the endpoint will be /ping.

def ping():
    return "pong!"

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=9696)  # This will run the Flask app on all available IP addresses on port 9696.


# curl is a command-line tool to communicate with web services.