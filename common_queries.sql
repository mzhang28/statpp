-- Top most impressive scores
select s.id, score, round(bm.success_rate, 4) as success, left(artist, 20) as artist, left(title, 20) as title, left(diffname, 20) as diffname, username, `mod`, round(bm.difficulty, 6) as difficulty, round(score_pp, 4) as pp
    from score as s
        join user as u on s.user = u.id
        join beatmapmod as bm on bm.id = s.beatmap_mod
        join beatmap as b on bm.beatmap = b.id
    where score_pp <> 0
    order by score_pp desc limit 30;

-- Top hardest maps
select round(difficulty, 6) as difficulty, round(bm.success_rate, 4) as success, `mod`, b.id, left(artist, 20) as artist, left(title, 20) as title, left(diffname, 20) as diffname from beatmap as b join beatmapmod as bm on b.id = bm.beatmap
    where difficulty <> 5
    order by difficulty desc limit 30;

-- Top top players
select id, username, round(total_pp, 2) as pp from user
    where total_pp <> 0
    order by total_pp desc limit 30;

-- Reset
update beatmapmod set difficulty = 5, success_rate = 0.0;
update score set score_pp = 0;
update user set total_pp = 0;
