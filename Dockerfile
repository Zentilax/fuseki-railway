# Use Java 17 slim image
FROM openjdk:17-jdk-slim

ENV FUSEKI_VERSION=5.5.0
ENV FUSEKI_HOME=/fuseki
ENV DATASET_NAME=ds
ENV DATA_PATH=/data

RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*
RUN curl -L https://downloads.apache.org/jena/binaries/apache-jena-fuseki-${FUSEKI_VERSION}.zip \
    -o fuseki.zip && \
    unzip fuseki.zip && \
    mv apache-jena-fuseki-${FUSEKI_VERSION} ${FUSEKI_HOME} && \
    rm fuseki.zip

WORKDIR ${FUSEKI_HOME}
COPY config.ttl ${FUSEKI_HOME}/config.ttl
COPY shiro.ini ${FUSEKI_HOME}/shiro.ini

EXPOSE 3030

#CMD ["java", "-Xmx1G", "-XX:-UseContainerSupport", "-jar", "fuseki-server.jar"]
CMD ["java", "-Xmx1G", "-XX:-UseContainerSupport", "-jar", "fuseki-server.jar", "--config=config.ttl"]

