FROM python:3.10-slim
WORKDIR /app
COPY test_model.py .
RUN pip install pytest
CMD ["pytest", "test_model.py"]