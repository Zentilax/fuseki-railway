# Use Java 17 slim image
FROM openjdk:17-jdk-slim

# Fuseki version
ENV FUSEKI_VERSION=5.5.0
ENV FUSEKI_HOME=/fuseki
ENV DATASET_NAME=ds
ENV DATA_PATH=/data

# Install curl & unzip, then download Fuseki
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*
RUN curl -L https://downloads.apache.org/jena/binaries/apache-jena-fuseki-${FUSEKI_VERSION}.zip \
    -o fuseki.zip && \
    unzip fuseki.zip && \
    mv apache-jena-fuseki-${FUSEKI_VERSION} ${FUSEKI_HOME} && \
    rm fuseki.zip

# Set working directory
WORKDIR ${FUSEKI_HOME}

# Expose Fuseki's port
EXPOSE 3030

# Start Fuseki with JVM flag to disable container support and set max memory
CMD ["java", "-Xmx1G", "-XX:+UseContainerSupport=false", "-jar", "fuseki-server.jar"]
