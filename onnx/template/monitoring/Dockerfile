FROM python:3.6.13-slim-buster
WORKDIR /usr/src/monitoring
ENV FLASK_APP=monitoring.py
ENV FLASK_RUN_HOST=0.0.0.0
# Server will reload itself on file changes if in dev mode
ENV FLASK_ENV=production
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run", "-p", "5001"]
