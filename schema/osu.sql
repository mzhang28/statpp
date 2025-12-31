-- MySQL dump 10.13  Distrib 9.3.0, for macos15.4 (arm64)
--
-- Host: 127.0.0.1    Database: osu
-- ------------------------------------------------------
-- Server version	12.0.2-MariaDB-ubu2404

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `osu_beatmap_difficulty`
--

DROP TABLE IF EXISTS `osu_beatmap_difficulty`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_beatmap_difficulty` (
  `beatmap_id` int(10) unsigned NOT NULL,
  `mode` tinyint(4) NOT NULL DEFAULT 0,
  `mods` int(10) unsigned NOT NULL,
  `diff_unified` float NOT NULL,
  `last_update` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`beatmap_id`,`mode`,`mods`),
  KEY `diff_sort` (`mode`,`mods`,`diff_unified`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_beatmap_difficulty_attribs`
--

DROP TABLE IF EXISTS `osu_beatmap_difficulty_attribs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_beatmap_difficulty_attribs` (
  `beatmap_id` mediumint(8) unsigned NOT NULL,
  `mode` tinyint(3) unsigned NOT NULL,
  `mods` int(10) unsigned NOT NULL,
  `attrib_id` tinyint(3) unsigned NOT NULL COMMENT 'see osu_difficulty_attribs table',
  `value` float DEFAULT NULL,
  PRIMARY KEY (`beatmap_id`,`mode`,`mods`,`attrib_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=COMPRESSED;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_beatmap_failtimes`
--

DROP TABLE IF EXISTS `osu_beatmap_failtimes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_beatmap_failtimes` (
  `beatmap_id` mediumint(9) NOT NULL,
  `type` enum('fail','exit') NOT NULL,
  `p1` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p2` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p3` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p4` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p5` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p6` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p7` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p8` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p9` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p10` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p11` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p12` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p13` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p14` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p15` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p16` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p17` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p18` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p19` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p20` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p21` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p22` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p23` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p24` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p25` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p26` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p27` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p28` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p29` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p30` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p31` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p32` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p33` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p34` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p35` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p36` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p37` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p38` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p39` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p40` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p41` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p42` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p43` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p44` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p45` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p46` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p47` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p48` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p49` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p50` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p51` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p52` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p53` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p54` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p55` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p56` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p57` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p58` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p59` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p60` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p61` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p62` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p63` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p64` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p65` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p66` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p67` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p68` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p69` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p70` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p71` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p72` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p73` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p74` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p75` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p76` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p77` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p78` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p79` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p80` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p81` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p82` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p83` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p84` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p85` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p86` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p87` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p88` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p89` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p90` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p91` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p92` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p93` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p94` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p95` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p96` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p97` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p98` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p99` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `p100` mediumint(8) unsigned NOT NULL DEFAULT 0,
  UNIQUE KEY `beatmap_id` (`beatmap_id`,`type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_uca1400_ai_ci ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_beatmap_performance_blacklist`
--

DROP TABLE IF EXISTS `osu_beatmap_performance_blacklist`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_beatmap_performance_blacklist` (
  `beatmap_id` int(10) unsigned NOT NULL,
  `mode` tinyint(3) unsigned NOT NULL,
  PRIMARY KEY (`beatmap_id`,`mode`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_beatmaps`
--

DROP TABLE IF EXISTS `osu_beatmaps`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_beatmaps` (
  `beatmap_id` mediumint(8) unsigned NOT NULL AUTO_INCREMENT,
  `beatmapset_id` mediumint(8) unsigned DEFAULT NULL,
  `user_id` int(10) unsigned NOT NULL DEFAULT 0,
  `filename` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `checksum` varchar(32) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT NULL,
  `version` varchar(80) CHARACTER SET latin1 COLLATE latin1_swedish_ci NOT NULL DEFAULT '',
  `total_length` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `hit_length` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `countTotal` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `countNormal` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `countSlider` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `countSpinner` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `diff_drain` float unsigned NOT NULL DEFAULT 0,
  `diff_size` float unsigned NOT NULL DEFAULT 0,
  `diff_overall` float unsigned NOT NULL DEFAULT 0,
  `diff_approach` float unsigned NOT NULL DEFAULT 0,
  `playmode` tinyint(3) unsigned NOT NULL DEFAULT 0,
  `approved` tinyint(4) NOT NULL DEFAULT 0,
  `last_update` timestamp NOT NULL DEFAULT current_timestamp(),
  `difficultyrating` float NOT NULL DEFAULT 0,
  `max_combo` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `playcount` int(10) unsigned NOT NULL DEFAULT 0,
  `passcount` int(10) unsigned NOT NULL DEFAULT 0,
  `youtube_preview` varchar(50) DEFAULT NULL,
  `score_version` tinyint(4) NOT NULL DEFAULT 1,
  `deleted_at` timestamp NULL DEFAULT NULL,
  `bpm` float NOT NULL DEFAULT 60,
  PRIMARY KEY (`beatmap_id`),
  KEY `beatmapset_id` (`beatmapset_id`),
  KEY `filename` (`filename`),
  KEY `checksum` (`checksum`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5143909 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_bin ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_beatmapsets`
--

DROP TABLE IF EXISTS `osu_beatmapsets`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_beatmapsets` (
  `beatmapset_id` mediumint(8) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` int(10) unsigned NOT NULL DEFAULT 0,
  `thread_id` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `artist` varchar(80) NOT NULL DEFAULT '',
  `artist_unicode` varchar(80) DEFAULT NULL,
  `title` varchar(80) NOT NULL DEFAULT '',
  `title_unicode` varchar(80) DEFAULT NULL,
  `creator` varchar(80) NOT NULL DEFAULT '',
  `source` varchar(200) NOT NULL DEFAULT '',
  `tags` varchar(1000) NOT NULL DEFAULT '',
  `video` tinyint(1) NOT NULL DEFAULT 0,
  `storyboard` tinyint(1) NOT NULL DEFAULT 0,
  `epilepsy` tinyint(1) NOT NULL DEFAULT 0,
  `bpm` float NOT NULL DEFAULT 0,
  `versions_available` tinyint(3) unsigned NOT NULL DEFAULT 1,
  `approved` tinyint(4) NOT NULL DEFAULT 0,
  `approvedby_id` int(10) unsigned DEFAULT NULL,
  `approved_date` timestamp NULL DEFAULT NULL,
  `submit_date` timestamp NULL DEFAULT NULL,
  `last_update` timestamp NOT NULL DEFAULT current_timestamp(),
  `filename` varchar(120) CHARACTER SET utf8mb3 COLLATE utf8mb3_bin DEFAULT NULL,
  `active` tinyint(1) NOT NULL DEFAULT 1,
  `rating` float unsigned NOT NULL DEFAULT 0,
  `offset` smallint(6) NOT NULL DEFAULT 0,
  `displaytitle` varchar(200) NOT NULL DEFAULT '',
  `genre_id` smallint(5) unsigned NOT NULL DEFAULT 1,
  `language_id` smallint(5) unsigned NOT NULL DEFAULT 1,
  `star_priority` smallint(6) NOT NULL DEFAULT 0,
  `filesize` bigint(20) NOT NULL DEFAULT 0,
  `filesize_novideo` bigint(20) DEFAULT NULL,
  `body_hash` binary(16) DEFAULT NULL,
  `header_hash` binary(16) DEFAULT NULL,
  `osz2_hash` binary(16) DEFAULT NULL,
  `download_disabled` tinyint(3) unsigned NOT NULL DEFAULT 0,
  `download_disabled_url` varchar(255) CHARACTER SET utf8mb3 COLLATE utf8mb3_bin DEFAULT NULL,
  `thread_icon_date` timestamp NULL DEFAULT NULL,
  `favourite_count` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `play_count` int(10) unsigned NOT NULL DEFAULT 0,
  `difficulty_names` varchar(2048) DEFAULT NULL,
  `cover_updated_at` timestamp NULL DEFAULT NULL,
  `discussion_enabled` tinyint(1) NOT NULL DEFAULT 0,
  `discussion_locked` tinyint(1) NOT NULL DEFAULT 0,
  `deleted_at` timestamp NULL DEFAULT NULL,
  `hype` int(11) NOT NULL DEFAULT 0,
  `nominations` int(11) NOT NULL DEFAULT 0,
  `previous_queue_duration` int(11) NOT NULL DEFAULT 0,
  `queued_at` timestamp NULL DEFAULT NULL,
  `storyboard_hash` varchar(32) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT NULL,
  `nsfw` tinyint(1) NOT NULL DEFAULT 0,
  `track_id` int(10) unsigned DEFAULT NULL,
  `spotlight` tinyint(1) NOT NULL DEFAULT 0,
  `comment_locked` tinyint(1) DEFAULT 0,
  `eligible_main_rulesets` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`eligible_main_rulesets`)),
  PRIMARY KEY (`beatmapset_id`),
  KEY `user_id` (`user_id`),
  KEY `thread_id` (`thread_id`),
  KEY `genre_id` (`genre_id`),
  KEY `approved_2` (`approved`,`star_priority`),
  KEY `approved` (`approved`,`active`,`approved_date`),
  KEY `favourite_count` (`favourite_count`),
  KEY `approved_3` (`approved`,`active`,`last_update`),
  KEY `filename` (`filename`),
  KEY `osu_beatmapsets_track_id_index` (`track_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2380116 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_uca1400_ai_ci ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_counts`
--

DROP TABLE IF EXISTS `osu_counts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_counts` (
  `name` varchar(200) NOT NULL,
  `count` bigint(20) unsigned NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_uca1400_ai_ci ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_difficulty_attribs`
--

DROP TABLE IF EXISTS `osu_difficulty_attribs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_difficulty_attribs` (
  `attrib_id` smallint(5) unsigned NOT NULL,
  `name` varchar(256) NOT NULL DEFAULT '',
  `visible` tinyint(1) NOT NULL DEFAULT 0,
  PRIMARY KEY (`attrib_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_scores_high`
--

DROP TABLE IF EXISTS `osu_scores_high`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_scores_high` (
  `score_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `beatmap_id` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `user_id` int(10) unsigned NOT NULL,
  `score` int(11) NOT NULL DEFAULT 0,
  `maxcombo` smallint(5) unsigned NOT NULL DEFAULT 0,
  `rank` enum('A','B','C','D','S','SH','X','XH') NOT NULL,
  `count50` smallint(5) unsigned NOT NULL DEFAULT 0,
  `count100` smallint(5) unsigned NOT NULL DEFAULT 0,
  `count300` smallint(5) unsigned NOT NULL DEFAULT 0,
  `countmiss` smallint(5) unsigned NOT NULL DEFAULT 0,
  `countgeki` smallint(5) unsigned NOT NULL DEFAULT 0,
  `countkatu` smallint(5) unsigned NOT NULL DEFAULT 0,
  `perfect` tinyint(1) NOT NULL DEFAULT 0,
  `enabled_mods` smallint(5) unsigned NOT NULL DEFAULT 0,
  `date` timestamp NOT NULL DEFAULT current_timestamp(),
  `pp` float DEFAULT NULL,
  `replay` tinyint(3) unsigned NOT NULL DEFAULT 0,
  `hidden` tinyint(4) NOT NULL DEFAULT 0,
  `country_acronym` char(2) CHARACTER SET utf8mb3 COLLATE utf8mb3_bin NOT NULL DEFAULT '',
  PRIMARY KEY (`score_id`),
  KEY `user_beatmap_rank` (`user_id`,`beatmap_id`,`rank`),
  KEY `beatmap_score_lookup_v2` (`beatmap_id`,`hidden`,`score` DESC,`score_id`)
) ENGINE=InnoDB AUTO_INCREMENT=4845945736 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_uca1400_ai_ci ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_user_beatmap_playcount`
--

DROP TABLE IF EXISTS `osu_user_beatmap_playcount`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_user_beatmap_playcount` (
  `user_id` int(10) unsigned NOT NULL DEFAULT 0,
  `beatmap_id` mediumint(8) unsigned NOT NULL,
  `playcount` smallint(5) unsigned NOT NULL,
  PRIMARY KEY (`user_id`,`beatmap_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_uca1400_ai_ci ROW_FORMAT=COMPRESSED;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `osu_user_stats`
--

DROP TABLE IF EXISTS `osu_user_stats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `osu_user_stats` (
  `user_id` int(10) unsigned NOT NULL,
  `count300` int(11) NOT NULL DEFAULT 0,
  `count100` int(11) NOT NULL DEFAULT 0,
  `count50` int(11) NOT NULL DEFAULT 0,
  `countMiss` int(11) NOT NULL DEFAULT 0,
  `accuracy_total` bigint(20) unsigned NOT NULL,
  `accuracy_count` bigint(20) unsigned NOT NULL,
  `accuracy` float NOT NULL,
  `playcount` mediumint(9) NOT NULL,
  `ranked_score` bigint(20) NOT NULL,
  `total_score` bigint(20) NOT NULL,
  `x_rank_count` mediumint(9) NOT NULL,
  `xh_rank_count` mediumint(9) DEFAULT 0,
  `s_rank_count` mediumint(9) NOT NULL,
  `sh_rank_count` mediumint(9) DEFAULT 0,
  `a_rank_count` mediumint(9) NOT NULL,
  `rank` mediumint(9) NOT NULL,
  `level` float unsigned NOT NULL,
  `replay_popularity` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `fail_count` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `exit_count` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `max_combo` smallint(5) unsigned NOT NULL DEFAULT 0,
  `country_acronym` char(2) NOT NULL DEFAULT '',
  `rank_score` float unsigned NOT NULL,
  `rank_score_index` int(10) unsigned NOT NULL,
  `accuracy_new` float unsigned NOT NULL,
  `last_update` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `last_played` timestamp NOT NULL DEFAULT current_timestamp(),
  `total_seconds_played` bigint(20) NOT NULL DEFAULT 0,
  PRIMARY KEY (`user_id`),
  KEY `ranked_score` (`ranked_score`),
  KEY `rank_score` (`rank_score`),
  KEY `country_acronym_2` (`country_acronym`,`rank_score`),
  KEY `playcount` (`playcount`),
  KEY `country_ranked_score` (`country_acronym`,`ranked_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_uca1400_ai_ci ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `scores`
--

DROP TABLE IF EXISTS `scores`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `scores` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` int(10) unsigned NOT NULL,
  `ruleset_id` smallint(5) unsigned NOT NULL,
  `beatmap_id` mediumint(8) unsigned NOT NULL,
  `has_replay` tinyint(1) NOT NULL DEFAULT 0,
  `preserve` tinyint(1) NOT NULL DEFAULT 0,
  `ranked` tinyint(1) NOT NULL DEFAULT 1,
  `rank` char(2) NOT NULL DEFAULT '',
  `passed` tinyint(4) NOT NULL DEFAULT 0,
  `accuracy` float NOT NULL DEFAULT 0,
  `max_combo` int(10) unsigned DEFAULT 0,
  `total_score` int(10) unsigned NOT NULL DEFAULT 0,
  `data` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`data`)),
  `pp` float unsigned DEFAULT NULL,
  `legacy_score_id` bigint(20) unsigned DEFAULT NULL,
  `legacy_total_score` int(10) unsigned NOT NULL DEFAULT 0,
  `started_at` timestamp NULL DEFAULT NULL,
  `ended_at` timestamp NOT NULL,
  `unix_updated_at` int(10) unsigned NOT NULL DEFAULT unix_timestamp(),
  `build_id` smallint(5) unsigned DEFAULT NULL,
  `random_key` double DEFAULT NULL,
  PRIMARY KEY (`id`,`preserve`,`unix_updated_at`),
  KEY `beatmap_user_index` (`beatmap_id`,`user_id`),
  KEY `legacy_score_lookup` (`ruleset_id`,`legacy_score_id`),
  KEY `user_ruleset_index` (`user_id`,`ruleset_id`,`pp`,`ranked`,`beatmap_id`,`accuracy`),
  KEY `user_recent` (`user_id`,`id`),
  KEY `idx_scores_random_key` (`random_key`)
) ENGINE=InnoDB AUTO_INCREMENT=4928637455 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=COMPRESSED
 PARTITION BY RANGE  COLUMNS(`preserve`,`unix_updated_at`)
(PARTITION `p20250530` VALUES LESS THAN (0,1748476800) ENGINE = InnoDB,
 PARTITION `p20250531` VALUES LESS THAN (0,1748563200) ENGINE = InnoDB,
 PARTITION `p20250601` VALUES LESS THAN (0,1748649600) ENGINE = InnoDB,
 PARTITION `p20250602` VALUES LESS THAN (0,1748736000) ENGINE = InnoDB,
 PARTITION `p20250603` VALUES LESS THAN (0,1748822400) ENGINE = InnoDB,
 PARTITION `p0catch` VALUES LESS THAN (0,MAXVALUE) ENGINE = InnoDB,
 PARTITION `p1` VALUES LESS THAN (MAXVALUE,MAXVALUE) ENGINE = InnoDB);
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-12-30 21:52:35
