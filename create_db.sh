docker-compose up -d
docker exec -t odqa-selectel-pgdocker-1 psql -U selectel -d selectel -a -f /usr/local/ODQA/create_table.sql
