FROM python:3

RUN mkdir -p /home/amf_ari
WORKDIR /home/amf_ari

RUN pip install --upgrade pip

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn

ADD app app
ADD boot.sh ./
RUN chmod +x boot.sh


ENV FLASK_APP app


EXPOSE 5000
ENTRYPOINT ["./boot.sh"]