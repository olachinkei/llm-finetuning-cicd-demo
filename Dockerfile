FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
COPY fine_tuning.py ./
COPY data_preparation.py ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "japanese-task-evaluation.py"]
