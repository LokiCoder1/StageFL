FROM fd-coordinator:5000/pytorch_image

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["flwr-serverapp"]