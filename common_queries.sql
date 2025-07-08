-- Top 30 most impressive scores
select artist, title, diffname, username, mods, score_pp from score as s join user as u on s.user = u.id join beatmap as b on s.beatmap = b.id
    where score_pp <> 0
    order by score_pp desc limit 30;

-- Top 30 hardest maps
select difficulty, id, artist, title, diffname from beatmap
    where difficulty <> 5
    order by difficulty desc limit 30;

-- Top 30 top players
select * from user
    where total_pp <> 0
    order by total_pp desc limit 30;

-- Reset
update beatmap set difficulty = 5;
update score set score_pp = 0;
update user set total_pp = 0;
