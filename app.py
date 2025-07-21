from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# --- USER AUTH FUNCTIONS ---
def create_user_table():
    conn = sqlite3.connect('users.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)')
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

def validate_user(username, password):
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = cur.fetchone()
    conn.close()
    return user

# --- MAIN ROUTES ---
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if validate_user(request.form['username'], request.form['password']):
            session['username'] = request.form['username']
            return redirect('/predict')
        else:
            return 'Invalid Credentials'
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        add_user(request.form['username'], request.form['password'])
        return redirect('/login')
    return render_template('signup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect('/login')

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        data = np.array(features).reshape(1, -1)
        prediction = model.predict(data)[0]

        reasons = get_reason(features)

        return render_template('index.html', prediction=prediction, reasons=reasons)

    return render_template('index.html')

def get_reason(values):
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    thresholds = {
        'age': 50,
        'trestbps': 130,
        'chol': 200,
        'fbs': 1,
        'thalach': 100,
        'oldpeak': 1.5
    }

    reasons = []
    for i, name in enumerate(feature_names):
        if name in thresholds and values[i] > thresholds[name]:
            reasons.append(f"{name} = {values[i]} is higher than normal threshold ({thresholds[name]})")

    if not reasons:
        return ["All input values are within normal range."]
    return reasons

if __name__ == '__main__':
    create_user_table()
    app.run(debug=True)
