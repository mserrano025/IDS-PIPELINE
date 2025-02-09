FROM openjdk:11-jdk

# Install python3 and pip3
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean;

# Using /app as the working directory
WORKDIR /app

# Copying the model_service.py file to the container
COPY model_service/model_service.py /app/model_service.py

# Copying the CICIDS2017 folder to the container
COPY CICIDS2017 /app/CICIDS2017

# Copying the models to the container
COPY models/final_model_Wednesday /app/models/final_model_Wednesday

# Installing the required libraries
RUN pip3 install flask pandas sklearn

RUN pip3 install h2o==3.46.0.5

# Command to run the model_service.py file
CMD ["python3", "model_service.py"]
