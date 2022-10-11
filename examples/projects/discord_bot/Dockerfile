FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/

ENV PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code

CMD ["python","/code/bot.py"]
