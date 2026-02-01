## Steps for importing data

we need both the maps, which will be used for running through the existing pp algo, as well as the scores db

1. go to https://data.ppy.sh/
2. download both the osu_files.tar.bz2 and the latest dump of top 10k
3. extract to `source-data` (not `training-data`, that will be for the parquet files)
   - obviously replace with ur specific details
   - `pv /path/to/2026_01_01_osu_files.tar | tar -C source-data -xzf -`
   - `pv /path/to/2026_01_01_performance_osu_top_10000.tar | pbzip2 -dc | tar -C source-data -xzOf - | mysql -uroot -proot -h127.0.0.1 --port 3306 osu`
