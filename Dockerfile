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
# Don't copy shiro.ini - run without authentication

EXPOSE 3030

# Run without authentication
CMD ["java", "-Xmx1G", "-XX:+UseContainerSupport", "-Djava.awt.headless=true", "-Dfile.encoding=UTF-8", "-Djetty.host=0.0.0.0", "-jar", "fuseki-server.jar", "--config=config.ttl"]
