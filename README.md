"# rakuten-mlops" 


Check connection Prometheus/Grafana
docker compose logs --tail=100 prometheus
docker compose logs --tail=100 grafana
docker compose exec grafana sh -c 'wget -qO- http://prometheus:9090/-/healthy'
