## Cross-lingual Open domain question answering
Cross-lingual Open domain question answering web demo. We are 2000000 articles from the English wikipedia on the first hundred tokens
### Requirements
Docker
`apt get install docker `
### Quick start 
`docker-compose up` \
After database starting up \
```docker exec -t odqa-selectel-pgdocker-1 psql -U selectel -d selectel -a -f /usr/local/ODQA/create_table.sql```