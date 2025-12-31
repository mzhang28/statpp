startdb:
    podman compose up -d

dump_schemas:
    mysqldump -uroot -proot -h127.0.0.1 -P3306 --no-data osu > schema/osu.sql
    mysqldump -uroot -proot -h127.0.0.1 -P3306 --no-data statpp > schema/statpp.sql

gen_sea_orm_entities:
    sea-orm-cli generate entity --database-url mysql://root:root@localhost:3306/statpp -o statpp-rs/src/entity
