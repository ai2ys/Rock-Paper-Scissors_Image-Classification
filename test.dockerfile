FROM python:3.11.7-alpine3.19

# Install packages/tools
RUN pip install --no-cache-dir \
    numpy==1.26.0 \
    Pillow==10.0.1 \
    requests==2.31.0

WORKDIR /app

