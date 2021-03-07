#Use python as base image
FROM python:3.7

RUN pip install --no-cache-dir --upgrade pip

#Use working directory /app
WORKDIR /app

#Copy all the content of current directory to /app
ADD . /app

#Installing required packages
RUN pip install --no-cache-dir -r requirements.txt

#Open port 5000
EXPOSE 5000

#Set environment variable
ENV NAME OpentoAll

#Run python program
CMD ["python","app.py"]
