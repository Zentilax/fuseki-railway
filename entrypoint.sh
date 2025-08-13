#!/bin/bash

# Start Jena Fuseki on internal port 3031 (avoiding conflict)
echo "🚀 Starting Jena Fuseki on internal port 3031..."
cd /fuseki
java \
    -Xmx1G \
    -XX:-UseContainerSupport \
    -Djava.awt.headless=true \
    -Dfile.encoding=UTF-8 \
    -Dfuseki.metrics.enabled=false \
    -Dfuseki.prometheus.enabled=false \
    -Dmicrometer.enabled=false \
    -Dio.micrometer.core.instrument.binder.system.FileDescriptorMetrics.enabled=false \
    -Dio.micrometer.core.instrument.binder.jvm.JvmMemoryMetrics.enabled=false \
    -Dio.micrometer.core.instrument.binder.system.ProcessorMetrics.enabled=false \
    -Dmanagement.metrics.enabled=false \
    -jar fuseki-server.jar \
    --port=3031 \
    --config=config.ttl &

# Wait for Fuseki to start
echo "⏳ Waiting for Fuseki to start..."
for i in {1..10}; do
    if curl -s http://localhost:3031/\$/ping >/dev/null; then
        echo "✅ Fuseki is up!"
        break
    else
        echo "⏳ Waiting for Fuseki to be ready... attempt $i"
        sleep 3
    fi
done


# Test Fuseki connection
echo "🔍 Testing Fuseki connection..."
cd /app
python3 -c "
import requests
import time
for i in range(5):
    try:
        r = requests.get('http://localhost:3031/\$/ping', timeout=2)
        if r.status_code == 200:
            print('✅ Fuseki is running on port 3031!')
            break
    except Exception as e:
        print(f'⏳ Attempt {i+1}/5 - waiting for Fuseki... {e}')
        time.sleep(3)
else:
    print('⚠️ Could not connect to Fuseki')
"

# Start Python API on port 3030 (Railway's expected port)
echo "🌐 Starting Python API with Fuseki proxy on port 5000..."
cd /app
export INTERNAL_FUSEKI_PORT=3031
python3 api_server.py