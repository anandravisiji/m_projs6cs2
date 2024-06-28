from flask import Flask, send_from_directory, request, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__, static_folder='static')

# Load the models
similarity = pickle.load(open('similarity.pkl', 'rb'))
new_df = pd.read_pickle('courses.pkl')

def recommend(course, new_df, similarity, num_recommendations=6):
    if course in new_df['course_name'].values:
        course_index = new_df[new_df['course_name'] == course].index[0]
    else:
        return []

    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
    recommended_courses = [new_df.iloc[i[0]].course_name for i in course_list]
    return recommended_courses

@app.route('/')
def home1():
    template_path = os.path.join(os.getcwd(), 'templete')
    print("Serving home1.html from:", template_path)
    return send_from_directory(template_path, 'home2.html')

@app.route('/recommend', methods=['POST'])
def recommend_course():
    data = request.json
    course_name = data['course']
    recommendations = recommend(course_name, new_df, similarity)
    return jsonify(recommendations)

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    app.run(debug=True)
