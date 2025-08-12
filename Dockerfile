# Use Java 11 instead - better Railway compatibility
FROM openjdk:11-jre-slim

ENV FUSEKI_VERSION=5.5.0
ENV FUSEKI_HOME=/fuseki
ENV DATA_PATH=/data

RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

RUN curl -L https://downloads.apache.org/jena/binaries/apache-jena-fuseki-${FUSEKI_VERSION}.zip \
    -o fuseki.zip && \
    unzip fuseki.zip && \
    mv apache-jena-fuseki-${FUSEKI_VERSION} ${FUSEKI_HOME} && \
    rm fuseki.zip

WORKDIR ${FUSEKI_HOME}

COPY config.ttl ${FUSEKI_HOME}/config.ttl

# Create the data directory
RUN mkdir -p /data && chmod 777 /data

EXPOSE 3030

# Simple command without container detection issues
CMD ["java", "-Xmx1G", "-Djava.awt.headless=true", "-Djetty.host=0.0.0.0", "-jar", "fuseki-server.jar", "--config=config.ttl"]
