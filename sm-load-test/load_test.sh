#. ~/sm-runtime/bin/activate

locust --locustfile=locustfile.py --worker &
locust --locustfile=locustfile.py --worker &
locust --locustfile=locustfile.py --worker &
locust --locustfile=locustfile.py --worker &
locust --locustfile=locustfile.py --worker &
locust --host=http://localhost:7080 --web-port 8089 --locustfile=locustfile.py --master 