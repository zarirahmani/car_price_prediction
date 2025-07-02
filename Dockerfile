FROM python:3.12-rc-slim-bullseye

RUN pip install pipenv

WORKDIR /app 
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]
COPY ["linear_regression_model.pkl", "./"]
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]