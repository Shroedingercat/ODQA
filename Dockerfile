FROM conda/miniconda3
CMD mkdir /usr/local/ODQA
WORKDIR /usr/local/ODQA

COPY ./Models /usr/local/ODQA/Models
COPY ./static /usr/local/ODQA/static
COPY ./templates /usr/local/ODQA/templates
COPY ./data /usr/local/ODQA/data
COPY app.py /usr/local/ODQA
COPY config.yaml /usr/local/ODQA
COPY requirements.txt /usr/local/ODQA

RUN pip install --upgrade pip
RUN conda install -c pytorch faiss-cpu
RUN apt-get update && apt-get -y install libpq-dev gcc
RUN pip install -r requirements.txt


ENTRYPOINT [ "python" ]
CMD [ "app.py" ]