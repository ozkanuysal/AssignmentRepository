FROM python:3.8-slim-buster
WORKDIR /app

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y libgomp1

RUN pip install --upgrade pip 

ADD . /app

#!TODO: change the refactor.py csv file path like : /app/marketing_campaign_data_socail_media.csv

COPY ./marketing_campaign_data_socail_media.csv /app/marketing_campaign_data_socail_media.csv

RUN pip install -r requirenments.txt

#port
EXPOSE 80
CMD ["python", "./refactor.py"]