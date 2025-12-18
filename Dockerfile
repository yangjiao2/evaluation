FROM python:3.10.8-slim-bullseye as build

ARG REGISTRY_TOKEN
ARG LLM_GATEWAY_CLIENT_ID
ARG LLM_GATEWAY_CLIENT_SECRET
ARG IT_SUPPORT_NVAUTH_ACCOUNT_TOKEN
ARG IT_SUPPORT_BEARER_TOKEN

#RUN apt-get update  \
#    && apt-get install libgomp1 \
#    && apt-get install build-essential -y  \
#    && python3 -m pip install --upgrade pip \
#    && apt-get update && apt-get install -y git

RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    git \
    curl && \
    python3 -m pip install --upgrade pip && \
    # below for rust with ddtrace experiments
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

# below for rust with ddtrace experiments
RUN /root/.cargo/bin/rustup update && /root/.cargo/bin/rustc --version

ENV PATH="/root/.cargo/bin:${PATH}"

RUN export HNSWLIB_NO_NATIVE=1

# Copy requirements.txt
COPY requirements.txt requirements.txt
# Install all required modules
RUN pip3 install -r requirements.txt
RUN pip3 list
## set working directory
WORKDIR /evalution

## add user
RUN addgroup --system nvbot && adduser --system --home /home/nvbot/ --group nvbot
RUN chown -R nvbot:nvbot /evalution && chmod -R 755 /evalution

## switch to non-root user
USER nvbot

# Copy all the files to fulfillments
COPY . /evalution

EXPOSE 5000
CMD ddtrace-run uvicorn main:app --workers 4 --host 0.0.0.0 --port 5000 --loop uvloop --timeout-keep-alive 300