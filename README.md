# Potato-disease-classification

docker run -t --rm -p 8502:8502 -v D:\ML-projects\Potato-disease:/Potato-disease tensorflow/serving --rest_api_port=8502 --model_config_file=/Potato-disease/models.config