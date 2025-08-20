# Use Java 17 slim image (matching your current setup)
FROM openjdk:17-jdk-slim

# Environment variables (matching your current setup)
ENV FUSEKI_VERSION=5.5.0
ENV FUSEKI_HOME=/fuseki
ENV DATASET_NAME=ds
ENV DATA_PATH=/data

# Install Python and system packages for FAISS
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Fuseki (matching your current setup)
RUN curl -L https://downloads.apache.org/jena/binaries/apache-jena-fuseki-${FUSEKI_VERSION}.zip \
    -o fuseki.zip && \
    unzip fuseki.zip && \
    mv apache-jena-fuseki-${FUSEKI_VERSION} ${FUSEKI_HOME} && \
    rm fuseki.zip

WORKDIR ${FUSEKI_HOME}

# Copy your existing config
COPY config.ttl ${FUSEKI_HOME}/config.ttl

# Copy Python requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy your Python scripts
COPY . /app/

# Create the data directory and ensure proper permissions (matching your setup)
RUN mkdir -p /data && chmod 777 /data

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

#fuseki
 
#flask api 
EXPOSE 5000   

ENTRYPOINT ["/entrypoint.sh"]