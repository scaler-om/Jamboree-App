FROM python:3.12-bookworm
WORKDIR /app-docker

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "-m", "flask", "--app", "flask_app.py", "run", "--host=0.0.0.0"]
