from flask import Flask, jsonify

app = Flask(__name__)

# Initialize an empty list to store values
values = []

@app.route('/')
def get_value():
    
    print("got request")

    if values:
        # Pop the first value from the list and return it
        value = values.pop(0)
        return value
    else:
        # Return a message if the list is empty
        return ""

@app.route('/add/<value>', methods=['GET'])
def add_value(value):
    # Append the new value to the end of the list
    values.append(value)
    return f'Value {value} added successfully'

if __name__ == '__main__':
    app.run(debug=True)
