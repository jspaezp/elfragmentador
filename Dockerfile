
FROM --platform=linux/amd64 python:3.10-slim
RUN apt update && apt install procps -y
LABEL MAINTAINER="J. Sebastian Paez"
COPY dist/*.whl /tmp/
WORKDIR /app
RUN WHEEL=$(ls /tmp/*.whl) && pip install ${WHEEL}
