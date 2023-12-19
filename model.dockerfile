FROM tensorflow/serving:2.14.0

ENV MODEL_NAME=rock_paper_scissors
# Install packages/tools
COPY models/rock_paper_scissors-saved_model /models/${MODEL_NAME}/1

# Start TensorFlow model server
CMD /usr/bin/tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=${MODEL_NAME} \
    --model_base_path=/models/${MODEL_NAME}