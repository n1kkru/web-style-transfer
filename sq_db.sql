CREATE TABLE IF NOT EXISTS users (
id integer PRIMARY KEY AUTOINCREMENT,
name text NOT NULL,
login text NOT NULL,
pass text NOT NULL,
time integer NOT NULL
);