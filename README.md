# Jamboree-App
Streamlit app and Flask app to access a student's probability of admission to Ivy League colleges

### Installation
Create a virtual environment and within the environment run the below command in the terminal \
```pip install -r requirements.txt ```

### Running the streamlit app
Run the below command in the terminal \
```streamlit run .\streamlit_app.py```

### Running the flask app
Run the below command in the terminal \
```flask --app .\flask_app.py run```

### Building the docker
Run the below command in the terminal \
```docker build -t admission_probability_app .``` 

### Running the docker
Run the below command in the terminal \
```docker run -p 8000:5000 -d admission_probability_app``` 
