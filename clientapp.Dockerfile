FROM damianocann/pytorch-flower-app:latest

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["flwr-clientapp"]