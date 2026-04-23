import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:3000/api',
});

export interface Player {
  id: number;
  username: string;
  metadata: {
    ratings: number[];
    [key: string]: any;
  };
}

export interface Beatmap {
  id: number;
  artist: string;
  title: string;
  version: string;
  metadata: {
    difficulties: number[];
    [key: string]: any;
  };
}

export interface Score {
  player_id: number;
  player_username: string;
  beatmap_id: number;
  mod_str: string;
  metadata: {
    scores: number[];
    accuracy: number;
    [key: string]: any;
  };
}

export const getDimensions = async () => {
  const { data } = await api.get<{ dimensions: number }>('/dimensions');
  return data.dimensions;
};

export const getMeta = async () => {
  const { data } = await api.get<Record<string, string>>('/meta');
  return data;
};

export const getTopPlayers = async (dim: number, limit = 50) => {
  const { data } = await api.get<Player[]>(`/players/top?dim=${dim}&limit=${limit}`);
  return data;
};

export const getHardestBeatmaps = async (dim: number, limit = 50) => {
  const { data } = await api.get<Beatmap[]>(`/beatmaps/hardest?dim=${dim}&limit=${limit}`);
  return data;
};

export const getBeatmap = async (id: number) => {
  const { data } = await api.get<Beatmap>(`/beatmaps/${id}`);
  return data;
};

export const getBeatmapScores = async (beatmapId: number, dim: number, limit = 50) => {
  const { data } = await api.get<Score[]>(`/beatmaps/${beatmapId}/scores?dim=${dim}&limit=${limit}`);
  return data;
};
