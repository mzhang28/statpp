The two databases are:

- osu: the export from osu db
- statpp: my db

Generally I'm trying to first extract osu db into the statpp db so I have a more minimal set of data to work with.

To run database commands:

```
podman compose exec db mariadb --password=root osu # for the osu database
podman compose exec db mariadb --password=root statpp # for the statpp database
```
