FROM alpine:latest
LABEL org.opencontainers.image.source https://github.com/SENERGY-Platform/analytics-operator-local-estimator-simple

ADD . /opt/app
WORKDIR /opt/app
RUN apk add --update python3 py3-numpy py3-pip
RUN pip install --no-cache-dir python-dateutil~=2.8.1 https://github.com/SENERGY-Platform/analytics-local-lib/archive/main.zip \
 --extra-index-url https://www.piwheels.org/simple
CMD [ "python3", "./main.py" ]