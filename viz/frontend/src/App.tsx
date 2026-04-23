import { useState, useEffect } from 'react';
import { useQuery, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { 
  createRootRoute, 
  createRoute, 
  createRouter, 
  RouterProvider, 
  Link, 
  Outlet, 
  useParams
} from '@tanstack/react-router';
import { Users, Trophy, Sun, Moon, Monitor, Hash, ArrowLeft, Star, GitBranch, Clock, Info, Search, X, ExternalLink } from 'lucide-react';
import { getDimensions, getTopPlayers, getHardestBeatmaps, getBeatmap, getBeatmapScores, getMeta, searchPlayers, getPlayer, getPlayerScores } from './api';
import type { Player, Beatmap, Score } from './api';
import { cn } from './lib/utils';

const queryClient = new QueryClient();

type Theme = 'light' | 'dark' | 'system';

// --- Router Setup ---

const rootRoute = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('theme') as Theme) || 'system';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const applyTheme = () => {
      const isDark = theme === 'dark' || (theme === 'system' && mediaQuery.matches);
      root.classList.toggle('dark', isDark);
    };

    applyTheme();
    localStorage.setItem('theme', theme);

    if (theme === 'system') {
      mediaQuery.addEventListener('change', applyTheme);
      return () => mediaQuery.removeEventListener('change', applyTheme);
    }
  }, [theme]);

  return (
    <div className="min-h-screen bg-white dark:bg-[#0a0a0a] text-zinc-900 dark:text-zinc-100 transition-colors duration-200 font-sans selection:bg-zinc-200 dark:selection:bg-zinc-800">
      <nav className="sticky top-0 z-50 border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-[#0a0a0a]/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-zinc-900 dark:bg-zinc-100 rounded flex items-center justify-center">
              <Hash className="w-5 h-5 text-white dark:text-black" />
            </div>
            <span className="font-semibold tracking-tight text-lg">statpp</span>
          </Link>

          <div className="flex items-center gap-2">
            {(['light', 'dark', 'system'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTheme(t)}
                className={cn(
                  "p-2 rounded-md transition-colors",
                  theme === t 
                    ? "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100" 
                    : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
                )}
              >
                {t === 'light' && <Sun className="w-4 h-4" />}
                {t === 'dark' && <Moon className="w-4 h-4" />}
                {t === 'system' && <Monitor className="w-4 h-4" />}
              </button>
            ))}
          </div>
        </div>
      </nav>
      <Outlet />
    </div>
  );
}

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Dashboard,
});

const beatmapRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/beatmap/$beatmapId',
  component: BeatmapPage,
});

const playerRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/player/$playerId',
  component: PlayerPage,
});

const routeTree = rootRoute.addChildren([indexRoute, beatmapRoute, playerRoute]);
const router = createRouter({ routeTree });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

// --- Components ---

function Dashboard() {
  const [dim, setDim] = useState(0);
  const [showDiff, setShowDiff] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  const { data: dimensions } = useQuery({
    queryKey: ['dimensions'],
    queryFn: getDimensions,
  });

  const { data: meta } = useQuery({
    queryKey: ['meta'],
    queryFn: getMeta,
  });

  const { data: searchResults, isLoading: searchLoading } = useQuery({
    queryKey: ['players', 'search', searchQuery, dim],
    queryFn: () => searchPlayers(searchQuery, dim),
    enabled: searchQuery.length > 1,
  });

  const { data: players, isLoading: playersLoading } = useQuery({
    queryKey: ['players', dim],
    queryFn: () => getTopPlayers(dim),
    enabled: dimensions !== undefined,
  });

  const { data: beatmaps, isLoading: beatmapsLoading } = useQuery({
    queryKey: ['beatmaps', dim],
    queryFn: () => getHardestBeatmaps(dim),
    enabled: dimensions !== undefined,
  });

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <header className="mb-12">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-8">
          <div className="flex flex-col gap-1">
            <h1 className="text-3xl font-bold tracking-tight">Dimensions Explorer</h1>
            <p className="text-zinc-500 dark:text-zinc-400">Slice and dice n-dimensional rating data with sub-second precision.</p>
          </div>
          
          {meta && (
            <div className="flex flex-wrap gap-4 text-[10px] font-mono text-zinc-400 border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="flex items-center gap-1.5">
                <Clock className="w-3 h-3" />
                <span>{new Date(meta.timestamp).toLocaleString()}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <GitBranch className="w-3 h-3" />
                <span title={meta.git_hash}>{meta.git_hash.slice(0, 7)}</span>
              </div>
              {meta.git_diff && meta.git_diff.trim().length > 0 && (
                <button 
                  onClick={() => setShowDiff(!showDiff)}
                  className="flex items-center gap-1.5 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
                >
                  <Info className="w-3 h-3" />
                  <span>{showDiff ? 'Hide Diff' : 'Show Diff'}</span>
                </button>
              )}
            </div>
          )}
        </div>

        {showDiff && meta?.git_diff && (
          <div className="mb-8 p-4 bg-zinc-900 text-zinc-300 rounded-lg overflow-x-auto text-[10px] font-mono leading-relaxed max-h-[300px] overflow-y-auto border border-zinc-800">
            <pre>{meta.git_diff}</pre>
          </div>
        )}
        
        {dimensions !== undefined && (
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: dimensions }).map((_, i) => (
              <button
                key={i}
                onClick={() => setDim(i)}
                className={cn(
                  "px-4 py-1.5 rounded-full text-sm font-medium transition-all border",
                  dim === i 
                    ? "bg-zinc-900 dark:bg-zinc-100 text-white dark:text-black border-transparent" 
                    : "bg-transparent border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:border-zinc-400 dark:hover:border-zinc-600"
                )}
              >
                Dimension {i}
              </button>
            ))}
          </div>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        <section className="space-y-4">
          <div className="flex items-center gap-2 px-1 justify-between">
            <div className="flex items-center gap-2">
              <Users className="w-5 h-5 text-zinc-400" />
              <h2 className="text-xl font-semibold">Players</h2>
            </div>
            
            <div className="relative group">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 group-focus-within:text-zinc-900 dark:group-focus-within:text-zinc-100 transition-colors" />
              <input 
                type="text"
                placeholder="Search players..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-9 py-1.5 rounded-full text-sm border border-zinc-200 dark:border-zinc-800 bg-transparent focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:focus:ring-zinc-100 focus:border-transparent transition-all w-48 focus:w-64"
              />
              {searchQuery && (
                <button 
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
          <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 w-16">Rank</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Player</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">Rating</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {searchQuery.length > 1 ? (
                  searchLoading ? (
                    <tr className="animate-pulse">
                      <td colSpan={3} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                    </tr>
                  ) : searchResults?.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="px-4 py-8 text-center text-zinc-500">No results found</td>
                    </tr>
                  ) : searchResults?.map((p: Player) => (
                    <tr key={p.id} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                      <td className="px-4 py-3 font-mono text-zinc-400 text-xs">
                        #{p.rank}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-between group/row">
                          <Link 
                            to="/player/$playerId" 
                            params={{ playerId: p.id.toString() }}
                            className="flex flex-col group/link"
                          >
                            <span className="font-semibold group-hover/link:underline">{p.username}</span>
                            <span className="text-[10px] text-zinc-400 font-mono">ID: {p.id}</span>
                          </Link>
                          {p.metadata.user_id && (
                            <a 
                              href={`https://osu.ppy.sh/users/${p.metadata.user_id}`}
                              target="_blank"
                              rel="noreferrer"
                              className="opacity-0 group-hover/row:opacity-100 p-1.5 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-all"
                              title="Open osu! profile"
                            >
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-bold tabular-nums text-lg">
                        {p.metadata.ratings[dim]?.toFixed(4)}
                      </td>
                    </tr>
                  ))
                ) : playersLoading ? (
                  [...Array(5)].map((_, i) => (
                    <tr key={i} className="animate-pulse">
                      <td colSpan={3} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                    </tr>
                  ))
                ) : players?.map((p: Player) => (
                  <tr key={p.id} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                    <td className="px-4 py-3 font-mono text-zinc-400 text-xs">
                      #{p.rank}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-between group/row">
                        <Link 
                          to="/player/$playerId" 
                          params={{ playerId: p.id.toString() }}
                          className="flex flex-col group/link"
                        >
                          <span className="font-semibold group-hover/link:underline">{p.username}</span>
                          <span className="text-[10px] text-zinc-400 font-mono">ID: {p.id}</span>
                        </Link>
                        {p.metadata.user_id && (
                          <a 
                            href={`https://osu.ppy.sh/users/${p.metadata.user_id}`}
                            target="_blank"
                            rel="noreferrer"
                            className="opacity-0 group-hover/row:opacity-100 p-1.5 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-all"
                            title="Open osu! profile"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-bold tabular-nums text-lg">
                      {p.metadata.ratings[dim]?.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex items-center gap-2 px-1">
            <Trophy className="w-5 h-5 text-zinc-400" />
            <h2 className="text-xl font-semibold">Hardest Beatmaps</h2>
          </div>
          <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Beatmap</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">Difficulty</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {beatmapsLoading ? (
                  [...Array(5)].map((_, i) => (
                    <tr key={i} className="animate-pulse">
                      <td colSpan={2} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                    </tr>
                  ))
                ) : beatmaps?.map((b: Beatmap) => (
                  <tr key={b.id} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-between group/row">
                        <Link 
                          to="/beatmap/$beatmapId" 
                          params={{ beatmapId: b.id.toString() }} 
                          className="flex flex-col group/link"
                        >
                          <span className="font-semibold group-hover/link:underline">{b.title}</span>
                          <span className="text-[10px] text-zinc-500">
                            {b.artist} <span className="text-zinc-400 dark:text-zinc-600">//</span> {b.version}
                          </span>
                        </Link>
                        <a 
                          href={`https://osu.ppy.sh/beatmaps/${b.title}`}
                          target="_blank"
                          rel="noreferrer"
                          className="opacity-0 group-hover/row:opacity-100 p-1.5 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 transition-all"
                          title="Open on osu! website"
                        >
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-bold tabular-nums text-lg text-red-500/80 dark:text-red-400/80">
                      {b.metadata.difficulties[dim]?.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}

function BeatmapPage() {
  const { beatmapId } = useParams({ from: '/beatmap/$beatmapId' });
  const [dim, setDim] = useState(0);

  const { data: beatmap, isLoading: beatmapLoading } = useQuery({
    queryKey: ['beatmap', beatmapId],
    queryFn: () => getBeatmap(parseInt(beatmapId)),
  });

  const { data: dimensions } = useQuery({
    queryKey: ['dimensions'],
    queryFn: getDimensions,
  });

  const { data: scores, isLoading: scoresLoading } = useQuery({
    queryKey: ['scores', beatmapId, dim],
    queryFn: () => getBeatmapScores(parseInt(beatmapId), dim),
    enabled: dimensions !== undefined,
  });

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <Link to="/" className="inline-flex items-center gap-2 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 mb-8 transition-colors group">
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
        <span className="text-sm font-medium">Back to Explorer</span>
      </Link>

      {beatmapLoading ? (
        <div className="h-40 w-full bg-zinc-100 dark:bg-zinc-800 animate-pulse rounded-lg mb-12" />
      ) : beatmap && (
        <header className="mb-12">
          <div className="flex flex-col gap-1 mb-6">
            <div className="flex items-start justify-between">
              <h1 className="text-4xl font-bold tracking-tight">{beatmap.title}</h1>
              <a 
                href={`https://osu.ppy.sh/beatmaps/${beatmap.title}`}
                target="_blank"
                rel="noreferrer"
                className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-md text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
                title="Open on osu! website"
              >
                <ExternalLink className="w-5 h-5" />
              </a>
            </div>
            <p className="text-lg text-zinc-500">
              {beatmap.artist} <span className="text-zinc-300 dark:text-zinc-700 mx-2">/</span> {beatmap.version}
            </p>
            <span className="text-xs font-mono text-zinc-400 mt-2">ID: {beatmap.id}</span>
          </div>
          
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            {beatmap.metadata.difficulties.map((d: number, i: number) => (
              <div key={i} className="bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 px-3 py-2.5 rounded-md">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Dim {i}</div>
                <div className="text-lg font-bold tabular-nums">{d.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </header>
      )}

      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-4">
          <div className="flex items-center gap-2">
            <Star className="w-5 h-5 text-zinc-400" />
            <h2 className="text-xl font-semibold">Leaderboard</h2>
          </div>

          {dimensions !== undefined && (
            <div className="flex gap-1">
              {Array.from({ length: dimensions }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => setDim(i)}
                  className={cn(
                    "px-3 py-1 rounded-md text-xs font-medium transition-all",
                    dim === i 
                      ? "bg-zinc-900 dark:bg-zinc-100 text-white dark:text-black" 
                      : "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
                  )}
                >
                  Dim {i}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Player</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Mods</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Score</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">Acc</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {scoresLoading ? (
                [...Array(5)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td colSpan={4} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                  </tr>
                ))
              ) : scores?.map((s: Score, idx: number) => (
                <tr key={idx} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                  <td className="px-4 py-3">
                    <Link 
                      to="/player/$playerId" 
                      params={{ playerId: s.player_id.toString() }}
                      className="flex flex-col group/link"
                    >
                      <span className="font-semibold group-hover/link:underline">{s.player_username}</span>
                      <span className="text-[10px] text-zinc-400 font-mono">ID: {s.player_id}</span>
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <span className="px-1.5 py-0.5 rounded border border-zinc-200 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 text-[10px] font-bold uppercase tabular-nums">
                      {s.mod_str}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-bold tabular-nums text-blue-600 dark:text-blue-400">
                    {s.metadata.scores[dim]?.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-zinc-500">
                    {(s.metadata.accuracy * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function PlayerPage() {
  const { playerId } = useParams({ from: '/player/$playerId' });
  const [dim, setDim] = useState(0);

  const { data: player, isLoading: playerLoading } = useQuery({
    queryKey: ['player', playerId],
    queryFn: () => getPlayer(parseInt(playerId)),
  });

  const { data: dimensions } = useQuery({
    queryKey: ['dimensions'],
    queryFn: getDimensions,
  });

  const { data: scores, isLoading: scoresLoading } = useQuery({
    queryKey: ['player-scores', playerId, dim],
    queryFn: () => getPlayerScores(parseInt(playerId), dim),
    enabled: dimensions !== undefined,
  });

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <Link to="/" className="inline-flex items-center gap-2 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 mb-8 transition-colors group">
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
        <span className="text-sm font-medium">Back to Explorer</span>
      </Link>

      {playerLoading ? (
        <div className="h-40 w-full bg-zinc-100 dark:bg-zinc-800 animate-pulse rounded-lg mb-12" />
      ) : player && (
        <header className="mb-12">
          <div className="flex flex-col gap-1 mb-6">
            <div className="flex items-start justify-between">
              <h1 className="text-4xl font-bold tracking-tight">{player.username}</h1>
              {player.metadata.user_id && (
                <a 
                  href={`https://osu.ppy.sh/users/${player.metadata.user_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-md text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
                  title="Open osu! profile"
                >
                  <ExternalLink className="w-5 h-5" />
                </a>
              )}
            </div>
            <div className="flex gap-4 mt-2">
              <span className="text-xs font-mono text-zinc-400">ID: {player.id}</span>
              <span className="text-xs font-mono text-zinc-400">Skill: {player.metadata.skill?.toFixed(2)}</span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            {player.metadata.ratings.map((r: number, i: number) => (
              <div key={i} className="bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 px-3 py-2.5 rounded-md">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Dim {i}</div>
                <div className="text-lg font-bold tabular-nums">{r.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </header>
      )}

      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-4">
          <div className="flex items-center gap-2">
            <Trophy className="w-5 h-5 text-zinc-400" />
            <h2 className="text-xl font-semibold">Top Scores</h2>
          </div>

          {dimensions !== undefined && (
            <div className="flex gap-1">
              {Array.from({ length: dimensions }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => setDim(i)}
                  className={cn(
                    "px-3 py-1 rounded-md text-xs font-medium transition-all",
                    dim === i 
                      ? "bg-zinc-900 dark:bg-zinc-100 text-white dark:text-black" 
                      : "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
                  )}
                >
                  Dim {i}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Beatmap</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Mods</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Score</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">Acc</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {scoresLoading ? (
                [...Array(5)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td colSpan={4} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                  </tr>
                ))
              ) : scores?.map((s: Score, idx: number) => (
                <tr key={idx} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                  <td className="px-4 py-3">
                    <Link 
                      to="/beatmap/$beatmapId" 
                      params={{ beatmapId: s.beatmap_id.toString() }}
                      className="flex flex-col group/link"
                    >
                      <span className="font-semibold group-hover/link:underline">{s.beatmap_title}</span>
                      <span className="text-[10px] text-zinc-500">
                        {s.beatmap_artist} <span className="text-zinc-400 dark:text-zinc-600">//</span> {s.beatmap_version}
                      </span>
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <span className="px-1.5 py-0.5 rounded border border-zinc-200 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 text-[10px] font-bold uppercase tabular-nums">
                      {s.mod_str}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-bold tabular-nums text-blue-600 dark:text-blue-400">
                    {s.metadata.scores[dim]?.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-zinc-500">
                    {(s.metadata.accuracy * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

export default App;
