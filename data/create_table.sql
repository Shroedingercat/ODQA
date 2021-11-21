CREATE TABLE wiki( id SERIAL, title text, text text);
COPY wiki(id, title, text) FROM '/usr/local/ODQA/docs.csv' DELIMITER ',' CSV HEADER;