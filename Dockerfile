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

# Create the data directory and ensure proper permissions
RUN mkdir -p /data && chmod 777 /data

EXPOSE 3030

# Disable ALL metrics and container detection for Railway
CMD ["java", \
    "-Xmx1G", \
    "-XX:-UseContainerSupport", \
    "-Djava.awt.headless=true", \
    "-Dfile.encoding=UTF-8", \
    "-Djetty.host=0.0.0.0", \
    "-Dfuseki.metrics.enabled=false", \
    "-Dfuseki.prometheus.enabled=false", \
    "-Dmicrometer.enabled=false", \
    "-Dio.micrometer.core.instrument.binder.system.FileDescriptorMetrics.enabled=false", \
    "-Dio.micrometer.core.instrument.binder.jvm.JvmMemoryMetrics.enabled=false", \
    "-Dio.micrometer.core.instrument.binder.system.ProcessorMetrics.enabled=false", \
    "-Dmanagement.metrics.enabled=false", \
    "-jar", "fuseki-server.jar", \
    "--config=config.ttl"]
