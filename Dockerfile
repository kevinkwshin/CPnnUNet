FROM pytorch/pytorch:latest

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Change ubuntu repository to Kakao mirror
RUN sed -i 's|http://[a-z]\+.ubuntu.com|https://mirror.kakao.com|g' /etc/apt/sources.list
# Configure python repository with Kakao mirror
RUN printf "%s\n"\
    "[global]"\
    "index-url=https://mirror.kakao.com/pypi/simple/"\
    "extra-index-url=https://pypi.org/simple/"\
    "trusted-host=mirror.kakao.com"\
    > /etc/pip.conf && pip install --no-cache-dir --root-user-action=ignore --upgrade pip

# Install essential packages
RUN apt-get update && apt-get install -y\
        tzdata\
        sudo\
        git\
        curl\
        vim\
        libgl1-mesa-glx\
        libglib2.0-0\
        libgtk2.0-dev\
        npm\
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt