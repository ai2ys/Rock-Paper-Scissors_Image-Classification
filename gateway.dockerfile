FROM python:3.11.7-alpine3.19

COPY gateway-requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip==23.3.2
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /app
COPY "gateway.py" ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "gateway:app"] 