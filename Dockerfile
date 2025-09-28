FROM --platform=linux/amd64 mcr.microsoft.com/azure-functions/python:4-python3.11
WORKDIR /home/site/wwwroot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
COPY . .