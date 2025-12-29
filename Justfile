startdb:
    podman compose up -d

dump_schemas:
    mysqldump -uroot -proot -h127.0.0.1 -P3306 --no-data osu > schema/osu.sql
    mysqldump -uroot -proot -h127.0.0.1 -P3306 --no-data statpp > schema/statpp.sql
