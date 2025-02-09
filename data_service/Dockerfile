FROM openjdk:11-jdk

# Install python3 and pip3
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean;

# Use /app as the working directory
WORKDIR /app

# Copying the data_service.py file to the container
COPY data_service/data_service.py /app/data_service.py

# Copying the CICIDS2017 folder to the container
COPY CICIDS2017 /app/CICIDS2017

RUN pip3 install flask pandas

RUN ls /app

CMD ["python3", "data_service.py"]
